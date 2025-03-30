import ale_py
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Union
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Box


from gymnasium.wrappers import TransformObservation, RecordEpisodeStatistics, ClipReward
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

class LazyFrames:
    """Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.

    Note:
        This object should only be converted to numpy array just before forward pass.
    """

    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames: list, lz4_compress: bool = False):
        """Lazyframe for a set of frames and if to apply lz4.

        Args:
            frames (list): The frames to convert to lazy frames
            lz4_compress (bool): Use lz4 to compress the frames internally

        Raises:
            DependencyNotInstalled: lz4 is not installed
        """
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            try:
                from lz4.block import compress
            except ImportError as e:
                raise DependencyNotInstalled(
                    "lz4 is not installed, run `pip install gymnasium[other]`"
                ) from e

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        """Gets a numpy array of stacked frames with specific dtype.

        Args:
            dtype: The dtype of the stacked frames

        Returns:
            The array of stacked frames with dtype
        """
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        """Returns the number of frame stacks.

        Returns:
            The number of frame stacks
        """
        return self.shape[0]

    def __getitem__(self, int_or_slice: Union[int, slice]):
        """Gets the stacked frames for a particular index or slice.

        Args:
            int_or_slice: Index or slice to get items for

        Returns:
            np.stacked frames for the int or slice

        """
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        """Checks that the current frames are equal to the other object."""
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )
        

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)

        [self.frames.append(obs) for _ in range(self.num_stack)]

        return self.observation(None), info


class MyFrameStack(gym.Wrapper):
    """
    A Gymnasium wrapper that stacks consecutive observations (frames) from an environment.

    This is commonly used in reinforcement learning for environments where a single
    observation does not contain enough information about the dynamics (e.g., velocity or acceleration).

    Args:
        env (gym.Env): The environment to wrap.
        num_stack (int): The number of frames to stack.
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, truncated, info

    def _get_observation(self):
        # Stack along the channel (first) axis: shape (4, 84, 84)
        return np.stack(self.frames, axis=0)


def make_env(env_name: str, render_mode: str = None):
    """Create and properly wrap an Atari environment."""
    # Create the environment with specified render mode and settings.
    env = gym.make(
        env_name,
        render_mode=render_mode,
        frameskip=1,  # Let AtariPreprocessing handle frame skipping
        full_action_space=False,
    )

    # Apply Atari-specific preprocessing.
    env = AtariPreprocessing(
        env,
        frame_skip=4,               # Skip 4 frames per action
        screen_size=84,             # Resize to 84x84
        grayscale_obs=True,         # Convert to grayscale
        terminal_on_life_loss=True, # End episode on life loss
    )

    # Define a new observation space with normalized pixels [0,1] as float32.
    original_obs_space = env.observation_space
    new_obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=original_obs_space.shape,
        dtype=np.float32,
    )

    # Normalize observations to [0,1] and convert to float32.
    env = TransformObservation(
        env,
        lambda obs: obs.astype(np.float32) / 255.0,
        observation_space=new_obs_space,
    )

    # Apply the FrameStack wrapper to stack 4 most recent frames.
    env = FrameStack(env, 4)
    
    
    # Clip rewards to {-1, 0, +1} for more stable learning.
    env = ClipReward(env, min_reward=-1, max_reward=1)
    
    # Record episode statistics such as cumulative rewards and episode length.
    env = RecordEpisodeStatistics(env)

    return env


