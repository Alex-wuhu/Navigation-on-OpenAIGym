
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
env = make_vec_env("CarRacing-v0")
env = VecFrameStack(env,n_stack=4)




model = PPO('CnnPolicy',env,verbose=0,tensorboard_log="log/PPO_tensorboard/")
eval_env = model.get_env()

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model.learn(total_timesteps=50000, tb_log_name="PPO_frame_4",callback=eval_callback)
model.save("gym_car/weights/PPO2_frame.pkl")

del model

env.close()