import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections
import itertools
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib import plotting

class ValueEstimator(nn.Module):
    """
    Value Function Approximator using a simple linear model with one-hot encoded states.

    This class estimates the value of discrete states. It takes the state index,
    converts it into a one-hot vector, and passes it through a single linear layer
    to produce a scalar value estimate. When bias is False in the linear layer,
    each weight directly corresponds to the estimated value of a specific state,
    making it functionally equivalent to a lookup table that can be updated via
    gradient descent.
    """

    def __init__(self, obs_space_n, learning_rate=0.1, device='cpu'):
        super(ValueEstimator, self).__init__()
        self.device = device

        # Simple linear layer: from one-hot state to scalar value
        self.model = nn.Sequential(
            nn.Linear(obs_space_n, 1, bias=False)
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _one_hot(self, state_index, num_classes):
        """Creates a one-hot encoded tensor for a discrete state index."""
        x = torch.zeros(num_classes, dtype=torch.float32)
        x[state_index] = 1.0
        return x.to(self.device)

    def predict(self, state):
        """
        Predict the value of a given state.

        Args:
            state: An integer representing the state index.

        Returns:
            Scalar value estimate (float).
        """
        with torch.no_grad():
            state_tensor = self._one_hot(state, self.model[0].in_features)
            value = self.model(state_tensor.unsqueeze(0))  # Shape: [1, 1]
        return value.item()

    def update(self, state, target):
        """
        Update the value function using a single (state, return) pair.

        Args:
            state: Integer state index.
            target: Target return (float).
        Returns:
            loss: Scalar loss value for this update step.
        """
        state_tensor = self._one_hot(state, self.model[0].in_features).unsqueeze(0)
        target_tensor = torch.tensor([[target]], dtype=torch.float32).to(self.device)

        self.model.train()
        prediction = self.model(state_tensor)
        loss = self.loss_fn(prediction, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class PolicyEstimator(nn.Module):
    """
    Policy Function Approximator for REINFORCE.
    
    Args:
        num_states (int): Number of states (e.g., env.observation_space.n).
        num_actions (int): Number of actions (e.g., env.action_space.n).
        learning_rate (float): Learning rate for the Adam optimizer.
    """
    
    def __init__(self, num_states, num_actions, learning_rate=0.01):
        super(PolicyEstimator, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Linear layer simulating a table lookup on one-hot encoded state.
        self.linear = nn.Linear(num_states, num_actions)
        
        # Initialize weights and biases to zeros (mimics tf.zeros_initializer).
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Adam optimizer for training.
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state):
        """
        Computes action probabilities for a given state.
        
        Args:
            state (torch.Tensor): A tensor containing the state index.
        
        Returns:
            torch.Tensor: A 1D tensor of action probabilities.
        """
        # Ensure state is on the same device as the model.
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.long, device=self.linear.weight.device)
        else:
            state = state.to(self.linear.weight.device)
        
        # Create one-hot encoding.
        one_hot = torch.zeros(self.num_states, device=state.device)
        one_hot[state] = 1.0
        
        # Compute logits with a batch dimension and obtain softmax probabilities.
        logits = self.linear(one_hot.unsqueeze(0))  # Shape: (1, num_actions)
        action_probs = F.softmax(logits, dim=1)
        return action_probs.squeeze()  # Shape: (num_actions,)
    
    def predict(self, state):
        """
        Predicts the action probability vector without computing gradients.
        
        Args:
            state (int or torch.Tensor): The state index.
        
        Returns:
            torch.Tensor: A tensor of action probabilities (summing to 1).
        """
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=torch.long, device=self.linear.weight.device)
            else:
                state = state.to(self.linear.weight.device)
            
            action_probs = self.forward(state)
            # Renormalize to counter any numerical issues (softmax already normalizes, but this is a safeguard).
            action_probs = action_probs / action_probs.sum()
            return action_probs.detach()

    def update(self, state, target, action):
        """
        Performs a gradient descent update on the policy for a given transition.
        
        Args:
            state (int or torch.Tensor): The current state.
            target (float): The advantage (or target) value.
            action (int): The action index taken.
        
        Returns:
            float: The computed scalar loss value.
        """
        self.train()
        self.optimizer.zero_grad()
        
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.long, device=self.linear.weight.device)
        else:
            state = state.to(self.linear.weight.device)
        
        # Compute action probabilities.
        action_probs = self.forward(state)
        picked_action_prob = action_probs[action]
        
        # Compute the loss: negative log likelihood weighted by the target (advantage).
        loss = -torch.log(picked_action_prob + 1e-8) * target
        
        # Backpropagate the loss and update the parameters.
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def batch_update(self, states, actions, advantages):
        self.train()
        self.optimizer.zero_grad()
        total_loss = 0.0

        for state, action, advantage in zip(states, actions, advantages):
            action_probs = self.forward(state)
            picked_action_prob = action_probs[action]
            loss = -torch.log(picked_action_prob + 1e-8) * advantage
            total_loss += loss

        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()


def reinforce(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    

    # Trajectory sampling
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset()
        
        episode = []
        
        for t in itertools.count():
            # One step in the environment
            action_probs = estimator_policy.predict(state)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{}, reward: ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done:
                break
                
            state = next_state
    
        # Go through the episode and make policy updates
        for t, transition in enumerate(episode):
            # Empirical return starting from timestep t
            total_return = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
            
            # Advantage estimation
            baseline_value = estimator_value.predict(transition.state)            
            advantage = total_return - baseline_value
            
            # Update our value estimator aka re-fit the baseline
            estimator_value.update(transition.state, total_return)
            # Update our policy estimator
            estimator_policy.update(transition.state, advantage, transition.action)

        # # After episode ends
        # states = []
        # actions = []
        # advantages = []

        # for t, transition in enumerate(episode):
        #     # Empirical return from timestep t
        #     total_return = sum(discount_factor**i * ep_step.reward for i, ep_step in enumerate(episode[t:]))
            
        #     # Predict baseline
        #     baseline_value = estimator_value.predict(transition.state)
        #     advantage = total_return - baseline_value

        #     # Collect for batch update
        #     states.append(transition.state)
        #     actions.append(transition.action)
        #     advantages.append(advantage)

        #     # Re-fit the value function to the actual return
        #     estimator_value.update(transition.state, total_return)

        # # Perform a single batch update for the whole episode
        # estimator_policy.batch_update(states, actions, advantages)
    
    return stats