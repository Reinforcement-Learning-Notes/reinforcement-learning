import os
from typing import Optional
import numpy as np
import torch as th

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from atari_env import make_env
from collect_data import collect_one_step, linear_schedule
from evaluation import evaluate_policy
from replay_buffer import ReplayBuffer
from q_network import CNNQNetwork

def dqn_update(
    q_net: CNNQNetwork,
    q_target_net: CNNQNetwork,
    optimizer: th.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: th.device,
) -> float:
    """
    Perform one gradient step on the Q-network
    using the data from the replay buffer.

    :param q_net: The Q-network to update
    :param q_target_net: The target Q-network, to compute the td-target.
    :param optimizer: The optimizer to use
    :param replay_buffer: The replay buffer containing the transitions
    :param batch_size: The minibatch size, how many transitions to sample
    :param gamma: The discount factor
    """

    # Sample the replay buffer and convert them to PyTorch tensors
    replay_data = replay_buffer.sample(batch_size).to_torch(device=device)

    with th.no_grad():
        # Use the target q-network instead of the online q-network
        # Compute the Q-values for the next observations (batch_size, n_actions)
        # using the target network
        next_q_values = q_target_net(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        # (batch_size,)
        next_q_values, _ = next_q_values.max(dim=1)
        # If the episode is terminated, set the target to the reward
        should_bootstrap = th.logical_not(replay_data.terminateds)
        # 1-step TD target
        td_target = replay_data.rewards + gamma * next_q_values * should_bootstrap

    # Get current Q-values estimates for the replay_data (batch_size, n_actions)
    q_values = q_net(replay_data.observations)
    # Select the Q-values corresponding to the actions that were selected during data collection
    current_q_values = th.gather(q_values, dim=1, index=replay_data.actions)
    # Reshape from (batch_size, 1) to (batch_size,) to avoid broadcast error
    current_q_values = current_q_values.squeeze(dim=1)

    # Check for any shape/broadcast error
    # Current q-values must have the same shape as the TD target
    assert current_q_values.shape == (batch_size,), f"{current_q_values.shape} != {(batch_size,)}"
    assert current_q_values.shape == td_target.shape, f"{current_q_values.shape} != {td_target.shape}"

    # Compute the Mean Squared Error (MSE) loss
    # Optionally, one can use a Huber loss instead of the MSE loss
    # loss = ((current_q_values - td_target) ** 2).mean()
    # Huber loss
    loss = th.nn.functional.smooth_l1_loss(current_q_values, td_target)

    # Reset gradients
    optimizer.zero_grad()
    # Compute the gradients
    loss.backward()
    # Update the parameters of the q-network
    optimizer.step()
    
    return loss.item()

def run_dqn(
    env_id: str = "ALE/Breakout-v5",
    replay_buffer_size: int = 50_000,
    target_network_update_interval: int = 1000,
    learning_starts: int = 100,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.01,
    exploration_fraction: float = 0.1,
    n_timesteps: int = 20_000,
    update_interval: int = 2,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    gamma: float = 0.99,
    n_eval_episodes: int = 10,
    evaluation_interval: int = 1000,
    eval_exploration_rate: float = 0.0,
    seed: int = 2023,
    eval_render_mode: Optional[str] = None,  # "human", "rgb_array", None
    device: Optional[str] = None, 
    checkpoint_path: Optional[str] = None,
    new_learning_rate: Optional[float] = None,
) -> CNNQNetwork:

    # Determine the device: GPU if available, otherwise CPU
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    device = th.device(device)
    print(f"Training on device: {device}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    th.manual_seed(seed)

    # Create TensorBoard writer
    log_dir = f"runs/dqn_{env_id.replace('/', '_')}_{seed}"
    writer = SummaryWriter(log_dir=log_dir)

    # Create the training environment using our custom make_env
    env = make_env(env_id, render_mode=None)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    env.action_space.seed(seed)

    # Create the evaluation environment using make_env
    eval_env = make_env(env_id, render_mode=eval_render_mode)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    # Create the Q-network and target network, and move them to device
    q_net = CNNQNetwork(env.observation_space, env.action_space).to(device)

    optimizer = th.optim.Adam(q_net.parameters(), lr=learning_rate)

    if checkpoint_path is not None:
        state_dict = th.load(checkpoint_path)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        q_net.load_state_dict(state_dict)
        
        # Optionally override the learning rate for fine-tuning
        if new_learning_rate is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate

    # Create the target network and move it to device
    q_target_net = CNNQNetwork(env.observation_space, env.action_space).to(device)
    q_target_net.load_state_dict(q_net.state_dict())

    # Create the Replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space)

    # Reset the environment
    obs, _ = env.reset(seed=seed)
    
    print("=== Training started ===")
    for current_step in range(1, n_timesteps + 1):
        if current_step % 10_000 == 0:
            print(f"Step {current_step}/{n_timesteps}")
        
        exploration_rate = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            current_step,
            int(exploration_fraction * n_timesteps),
        )
        
        if current_step % 100 == 0:
            writer.add_scalar("train/exploration_rate", exploration_rate, current_step)

        # Collect one step using epsilon-greedy policy
        obs = collect_one_step(
            env,
            q_net,
            replay_buffer,
            obs,
            exploration_rate=exploration_rate,
            verbose=0,
        )

        # Update the target network
        # by copying the parameters from the Q-network every target_network_update_interval steps
        if (current_step % target_network_update_interval) == 0:
            q_target_net.load_state_dict(q_net.state_dict())

        # Update the Q-network every update_interval steps
        # after learning_starts steps have passed (warmup phase)
        if (current_step % update_interval) == 0 and current_step > learning_starts:
            loss = dqn_update(q_net, q_target_net, optimizer, replay_buffer, batch_size, gamma=gamma, device=device)
            writer.add_scalar("train/loss", loss, current_step)
            
            # Log average Q-value over a sampled batch
            replay_data = replay_buffer.sample(batch_size).to_torch(device=device)
            with th.no_grad():
                q_vals = q_net(replay_data.observations)
                avg_q_val = q_vals.mean().item()
            writer.add_scalar("train/avg_q_value", avg_q_val, current_step)

        if (current_step % evaluation_interval) == 0:
            print(f"\n=== Evaluation at step {current_step} ===")
            print(f"Exploration rate: {exploration_rate:.2f}")
            mean_reward, std_reward = evaluate_policy(eval_env, q_net, n_eval_episodes, eval_exploration_rate=eval_exploration_rate)
            writer.add_scalar("eval/mean_episode_reward", mean_reward, current_step)
            writer.add_scalar("eval/std_episode_reward", std_reward, current_step)
            
            safe_env_id = env_id.replace("/", "_")
            os.makedirs("./logs", exist_ok=True)
            th.save(q_net.state_dict(), f"./logs/q_net_checkpoint_{safe_env_id}_{current_step}.pth")
            
    print("=== Training finished ===")
    writer.close()
    return q_net