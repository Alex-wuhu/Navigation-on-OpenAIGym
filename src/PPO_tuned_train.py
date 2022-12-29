from unittest.mock import call
from stable_baselines3 import PPO
import gym
from typing import Callable
import torch.nn as nn


from stable_baselines3.common.callbacks import EvalCallback
from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
from gym.wrappers import NormalizeReward

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
env = gym.make("CarRacing-v0")

env = GrayScaleObservation(env,keep_dim=True)


env = ResizeObservation(env,shape=64)
env = NormalizeReward(env)




model =PPO("CnnPolicy",env,verbose=0, tensorboard_log="log/PPO_tensorboard/",
        batch_size=128, n_steps=512, gamma=0.99, n_epochs=10,
        gae_lambda=0.95, ent_coef=0.0, sde_sample_freq=4,
        max_grad_norm=0.5, vf_coef= 0.5, learning_rate=linear_schedule(1e-4),
        use_sde=True , clip_range=0.2 ,
        policy_kwargs= dict(log_std_init=-2, ortho_init=False, activation_fn=nn.GELU, net_arch=[dict(pi=[256],vf=[256])]) 
    )

eval_env = model.get_env()

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/PPO_new',
                             log_path='./logs/PPO_new', eval_freq=5000,
                             deterministic=True, render=False)


model.learn(total_timesteps=200000 , tb_log_name="PPO3_tuned_log",callback=eval_callback)


model.save("gym_car/weights/PPO3_tuned.pkl")
del model

env.close()