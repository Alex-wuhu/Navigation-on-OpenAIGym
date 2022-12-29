
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
env = make_vec_env("CarRacing-v0")

model =PPO("CnnPolicy",env,verbose=0, tensorboard_log="log/PPO_para/")

model.learn(total_timesteps=30000 , tb_log_name="PPO3_log")

model.save("gym_car/weights/PPO3_para.pkl")
del model

env.close()