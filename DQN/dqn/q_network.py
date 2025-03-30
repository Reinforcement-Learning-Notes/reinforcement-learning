import torch.nn as nn
import torch as th
from gymnasium import spaces
from typing import Type


class QNetwork(nn.Module):
    """
    A Q-Network for the DQN algorithm
    to estimate the q-value for a given observation.

    :param observation_space: Observation space of the env,
        contains information about the observation type and shape.
    :param action_space: Action space of the env,
        contains information about the number of actions.
    :param n_hidden_units: Number of units for each hidden layer.
    :param activation_fn: Activation function (ReLU by default)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        # Assume 1d space
        obs_dim = observation_space.shape[0]
        # Retrieve the number of discrete actions
        n_actions = int(action_space.n)
        # Create the q network (2 fully connected hidden layers)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of observation (batch_size, obs_dim)
        :return: The Q-values for the given observations
            for all the action (batch_size, n_actions)
        """
        return self.q_net(observations)

class CNNQNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: spaces.Box, 
        action_space: spaces.Discrete, 
        activation_fn: Type[nn.Module] = nn.ReLU):
        """
        Args:
            observation_space: Gymnasium observation space (must be Box)
            action_space: Gymnasium action space (must be Discrete)
            activation_fn: The activation function to use in the network
        """
        """Initialize the CNN Q-network."""
        super().__init__()  # Simplified super() call
        
        # Extract shape and number of actions
        input_shape = observation_space.shape
        num_actions = action_space.n
        
        # Check input shape is (C, H, W)
        if len(input_shape) != 3:
            raise ValueError(f"Observation space shape must be (C, H, W), got {input_shape}")
        
        in_channels = input_shape[0]

        # Build the Q-network
        self.q_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            activation_fn(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            activation_fn(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            activation_fn(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            activation_fn(),
            nn.Linear(512, num_actions)
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _get_conv_output(self, shape):
        """
        Helper function to calculate the output size of the convolutional layers.
        """
        dummy_input = th.zeros(1, *shape)
        with th.no_grad():
            return self.q_net[0](dummy_input).shape[1]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass with device checking
        if not x.is_cuda and next(self.parameters()).is_cuda:
            x = x.to('cuda')
        return self.q_net(x)
