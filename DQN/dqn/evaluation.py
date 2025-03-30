import warnings
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from collect_data import epsilon_greedy_action_selection
from q_network import CNNQNetwork


def evaluate_policy(
    eval_env: gym.Env,
    q_net: CNNQNetwork,
    n_eval_episodes: int,
    eval_exploration_rate: float = 0.0,
    video_name: Optional[str] = None,
) -> None:
    """
    Evaluate the policy by computing the average episode reward
    over n_eval_episodes episodes.
    Record video for each episode if video_name is set.

    :param eval_env: The environment to evaluate the policy on
    :param q_net: The Q-network to evaluate
    :param n_eval_episodes: The number of episodes to evaluate the policy on
    :param eval_exploration_rate: The exploration rate to use during evaluation
    :param video_name: When set, the filename of the video to record.
    """
    # Setup video recording if requested
    if video_name is not None and eval_env.render_mode == "rgb_array":
        video_folder = Path.cwd() / "logs/videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        
        # Silence warnings about existing folders
        warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")
        
        # Wrap the environment for video recording
        eval_env = RecordVideo(
            env=eval_env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,  # Record all episodes
            name_prefix=video_name,
            disable_logger=True,  # Disable internal logging
        )
        
    # Setup device
    device = next(q_net.parameters()).device
    episode_returns = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        
        while not done:
            action = epsilon_greedy_action_selection(
                q_net,
                obs,
                exploration_rate=eval_exploration_rate,
                action_space=eval_env.action_space,
                device=device,
            )
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        
        episode_returns.append(total_reward)

    # Print evaluation results
    print(f"Mean episode reward: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
    
    # Close the environment (this will finalize video recording)
    eval_env.close()
    
    return np.mean(episode_returns), np.std(episode_returns)