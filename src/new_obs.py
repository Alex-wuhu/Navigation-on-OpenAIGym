import gym
from gym import ObservationWrapper
from gym.wrappers import GrayScaleObservation
from gym.wrappers import FrameStack
from gym.wrappers import ResizeObservation
from gym.wrappers import NormalizeReward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env("CarRacing-v0")

#env = GrayScaleObservation(env,keep_dim=True)

env = VecFrameStack(env,n_stack=2)

env = ResizeObservation(env,shape=64)
env = NormalizeReward(env)
check_env(env)
