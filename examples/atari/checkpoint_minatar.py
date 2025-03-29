import argparse
import datetime
import os
import pprint
import sys
import json

import numpy as np
import torch
from atari_network import MinatarRainbow
from atari_wrapper import make_atari_env

from tianshou.data import (
    Collector,
    CollectStats,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import C51Policy, RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer

import gymnasium as gym
from tianshou.utils.space_info import SpaceInfo
from tianshou.env import DummyVectorEnv, SubprocVectorEnv

class ChannelFirstWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(old_shape[2], old_shape[0], old_shape[1]),  # HWC -> CHW
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info 

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._convert_obs(obs), reward, term, trunc, info

    def _convert_obs(self, obs):
        return np.transpose(obs, (2, 0, 1))

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="MinAtar/Breakout-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=-10.0)
    parser.add_argument("--v-max", type=float, default=10.0)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Steps between saving checkpoints")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    return parser.parse_args()

def get_checkpoint_dir(args, now=None):
    """Get the directory for checkpoints and completion flag."""
    if now is None:
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    return log_path

def get_completion_flag_path(checkpoint_dir):
    """Get the path to the completion flag file."""
    return os.path.join(checkpoint_dir, "training_completed.json")

def is_training_completed(checkpoint_dir):
    """Check if training was already completed."""
    completion_flag_path = get_completion_flag_path(checkpoint_dir)
    if os.path.exists(completion_flag_path):
        with open(completion_flag_path, 'r') as f:
            completion_data = json.load(f)
            return completion_data.get('completed', False)
    return False

def mark_training_completed(checkpoint_dir, stats=None):
    """Mark the training as completed."""
    completion_flag_path = get_completion_flag_path(checkpoint_dir)
    completion_data = {
        'completed': True,
        'completion_time': datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        'stats': stats if stats else {}
    }
    os.makedirs(os.path.dirname(completion_flag_path), exist_ok=True)
    with open(completion_flag_path, 'w') as f:
        json.dump(completion_data, f)

def get_checkpoint_path(checkpoint_dir):
    """Get the path to the checkpoint file."""
    return os.path.join(checkpoint_dir, "checkpoint.pth")

def save_checkpoint(policy, optim, train_collector, epoch, env_step, checkpoint_dir):
    """Save a checkpoint of the current training state."""
    checkpoint_path = get_checkpoint_path(checkpoint_dir)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'policy': policy.state_dict(),
        'optimizer': optim.state_dict(),
        'epoch': epoch,
        'env_step': env_step,
    }
    
    # For buffer, we don't save it directly to avoid memory issues
    # Instead, we'll track progress and training state
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at step {env_step}")

def test_rainbow(args: argparse.Namespace = get_args()) -> None:
    # Set the algorithm name
    args.algo_name = "rainbow"
    
    # Get timestamp for this run
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    
    # Set up checkpoint directory and paths
    checkpoint_dir = get_checkpoint_dir(args, now)
    
    # Check if we have a previous checkpoint to resume from
    previous_checkpoints = []
    if args.resume_path is None:
        # Look for existing checkpoints with the same task, algo, and seed
        base_dir = os.path.join(args.logdir, args.task, args.algo_name, str(args.seed))
        if os.path.exists(base_dir):
            for timestamp_dir in os.listdir(base_dir):
                timestamp_path = os.path.join(base_dir, timestamp_dir)
                if os.path.isdir(timestamp_path):
                    checkpoint_path = os.path.join(timestamp_path, "checkpoint.pth")
                    if os.path.exists(checkpoint_path):
                        previous_checkpoints.append((timestamp_path, checkpoint_path))
    
    # Sort by modification time to get the most recent checkpoint
    previous_checkpoints.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    
    # Check if most recent training was already completed
    if previous_checkpoints:
        most_recent_dir, most_recent_checkpoint = previous_checkpoints[0]
        if is_training_completed(most_recent_dir):
            print(f"Training was already completed in {most_recent_dir}. Skipping training.")
            # You might want to perform evaluation here
            args.watch = True
            args.resume_path = os.path.join(most_recent_dir, "policy.pth")
            checkpoint_dir = most_recent_dir
    
    if "MinAtar" in args.task:
        args.frames_stack = 1
        args.save_only_last_obs = False
        args.ignore_obs_next = False

    env = ChannelFirstWrapper(gym.make(args.task))
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    train_envs = DummyVectorEnv([lambda: ChannelFirstWrapper(gym.make(args.task)) for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: ChannelFirstWrapper(gym.make(args.task)) for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    
    # define model
    net = MinatarRainbow(
        *args.state_shape,
        args.action_shape,
        args.num_atoms,
        args.noisy_std,
        args.device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # define policy
    policy: C51Policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        action_space=env.action_space,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)
    
    # Create replay buffer
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True, 
            save_only_last_obs=True,
            stack_num=args.frames_stack,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=args.ignore_obs_next,
            save_only_last_obs=args.save_only_last_obs,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm,
        )
    
    # Handle checkpoint loading
    start_epoch = 0
    env_step = 0
    
    # Load from a specific resume path (if specified) or most recent checkpoint
    resume_path = args.resume_path
    if resume_path is None and previous_checkpoints:
        resume_path = previous_checkpoints[0][1]
        checkpoint_dir = previous_checkpoints[0][0]
        print(f"Using most recent checkpoint: {resume_path}")
    
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=args.device)
        
        # Check if it's a checkpoint or just a policy file
        if isinstance(checkpoint, dict) and 'policy' in checkpoint:
            # It's a full checkpoint
            policy.load_state_dict(checkpoint['policy'])
            optim.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)
            env_step = checkpoint.get('env_step', 0)
            print(f"Resuming from epoch {start_epoch}, step {env_step}")
        else:
            # It's just a policy file
            policy.load_state_dict(checkpoint)
            print("Loaded policy weights only.")
    
    # If watching only, skip training
    if args.watch:
        # Define test function
        def watch() -> None:
            print("Setup test envs ...")
            policy.set_eps(args.eps_test)
            test_envs.seed(args.seed)
            if args.save_buffer_name:
                print(f"Generate buffer with size {args.buffer_size}")
                buffer = PrioritizedVectorReplayBuffer(
                    args.buffer_size,
                    buffer_num=len(test_envs),
                    ignore_obs_next=True,
                    save_only_last_obs=True,
                    stack_num=args.frames_stack,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                collector = Collector[CollectStats](policy, test_envs, buffer, exploration_noise=True)
                result = collector.collect(n_step=args.buffer_size, reset_before_collect=True)
                print(f"Save buffer into {args.save_buffer_name}")
                # Unfortunately, pickle will cause oom with 1M buffer size
                buffer.save_hdf5(args.save_buffer_name)
            else:
                print("Testing agent ...")
                test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)
                test_collector.reset()
                result = test_collector.collect(n_episode=args.test_num, render=args.render)
            result.pprint_asdict()
                
        watch()
        return

    # collector
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs, exploration_noise=True)

    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"

    # Use existing run ID if resuming
    resume_id = args.resume_id
    if resume_path and not resume_id:
        # Try to extract resume_id from the path
        path_parts = os.path.normpath(resume_path).split(os.sep)
        if len(path_parts) >= 4:  # Should have at least task/algo/seed/timestamp
            resume_id = path_parts[-2]  # Use the timestamp as run ID

    # Create logger with proper resume ID
    logger = logger_factory.create_logger(
        log_dir=checkpoint_dir,
        experiment_name=os.path.basename(checkpoint_dir),
        run_id=resume_id,
        config_dict=vars(args),
    )

    def save_best_fn(policy: BasePolicy) -> None:
        """Save best policy."""
        torch.save(policy.state_dict(), os.path.join(checkpoint_dir, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        """Stop training when reaching goal."""
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in args.task:
            return mean_rewards >= 20
        return False

    def train_fn(epoch: int, env_step: int) -> None:
        """Adjust epsilon and beta during training."""
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})
                
        # Save checkpoint periodically
        if env_step % args.checkpoint_freq == 0 and env_step > 0:
            save_checkpoint(policy, optim, train_collector, epoch, env_step, checkpoint_dir)

    def test_fn(epoch: int, env_step: int | None) -> None:
        """Set epsilon for testing."""
        policy.set_eps(args.eps_test)
    
    # Custom save_checkpoint function for use with the trainer
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> None:
        """Checkpoint saver called by trainer."""
        if env_step % args.checkpoint_freq == 0 and env_step > 0:
            save_checkpoint(policy, optim, train_collector, epoch, env_step, checkpoint_dir)

    # test train_collector and start filling replay buffer
    train_collector.reset()
    
    # If resuming training, we might want to skip the initial buffer filling
    if env_step == 0:
        train_collector.collect(n_step=args.batch_size * args.training_num)

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=logger,
        test_in_train=False,
        resume_from_log=bool(resume_path),
    ).run()

    # Mark training as completed
    mark_training_completed(checkpoint_dir, stats={
        'returns_mean': float(result['best_reward']),
        'returns_std': float(result.get('best_reward_std', 0)),
        'length_mean': float(result.get('best_length', 0)),
        'train_time': result.get('duration', 0),
    })
    
    pprint.pprint(result)
    
    # Run evaluation with the final policy
    def watch() -> None:
        print("Testing agent ...")
        policy.set_eps(args.eps_test)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num, render=args.render)
        result.pprint_asdict()
    
    watch()


if __name__ == "__main__":
    test_rainbow(get_args())