
from subprocess import call
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback

env = make_vec_env("CarRacing-v0")



model =PPO("CnnPolicy",env,verbose=0, tensorboard_log="log/PPO/")


eval_env = model.get_env()

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/continu',
                             log_path='./logs/continu', eval_freq=5000,
                             deterministic=True, render=False)
model.learn(total_timesteps=30000, tb_log_name="first_run",callback=eval_callback)
# Pass reset_num_timesteps=False to continue the training curve in tensorboard
# By default, it will create a new curve
# Keep tb_log_name constant to have continuous curve (see note below)
model.learn(total_timesteps=30000, tb_log_name="second_run", reset_num_timesteps=False,callback=eval_callback)
model.learn(total_timesteps=30000, tb_log_name="third_run", reset_num_timesteps=False,callback=eval_callback)



model.save("gym_car/weights/PPO1_c.pkl")
del model

env.close()