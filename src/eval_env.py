
from email import parser
import argparse
from statistics import mode
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
from stable_baselines3.common.vec_env import VecFrameStack
import gym

if __name__=='__main__':
    #set parameters
    eval_env = gym.make("CarRacing-v0")
    #eval_env = VecFrameStack(eval_env, n_stack=4)
    parser=argparse.ArgumentParser(description='Play Car by model.')
    parser.add_argument('-w', '--weights', default='car_racing_weights1.pkl', help='The pkl file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=5, help='The number of episodes should the model plays.')
    args=parser.parse_args()
    #weight='weights\\'+args.weights
    
    eval_env = ResizeObservation(eval_env,shape=64)
    
    eval_env = GrayScaleObservation(eval_env,keep_dim=True)
    weight= "../logs/PPO_new/best_model.zip"

    episodes=args.episodes
    model = PPO.load(weight)
    #model = DQN.load(weight)
    #model = A2C.load(weight)

    #eval_env = model.get_env()

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=episodes,render=True)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

