from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
env = make_vec_env("CarRacing-v0")

model =PPO("CnnPolicy",env,verbose=0, tensorboard_log="log/PPO_nrf/")

eval_env = model.get_env()

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/PPO_nrf2',
                             log_path='./logs/PPO_nrf2', eval_freq=5000,
                             deterministic=True, render=False)

model.learn(total_timesteps=50000 , tb_log_name="first_run",callback=eval_callback)

model.learn(total_timesteps=50000 , tb_log_name="second_run",callback=eval_callback,reset_num_timesteps=False)

model.learn(total_timesteps=50000 , tb_log_name="third_run",callback=eval_callback,reset_num_timesteps=False)

model.learn(total_timesteps=50000 , tb_log_name="forth_run",callback=eval_callback,reset_num_timesteps=False)

model.save("gym_car/weights/PPO6_nrf.pkl")
del model

env.close()