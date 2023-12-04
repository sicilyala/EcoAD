import gymnasium as gym
from stable_baselines3 import DDPG
from tqdm import tqdm
from matplotlib import pyplot as plt

from highway_env import register_highway_envs
from arguments import get_args
from env_config import get_config
from utils import print_obs, print_info


if __name__=='__main__':
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make('EcoAD-v0', render_mode='rgb_array', config=config) 
    dir_name = "test_EMS/DDPG_1"
    log_dir = "./EcoHighway_DRL/" + dir_name

    
    print("\n----------Start Evaluating----------")
    DRL_agent = DDPG.load(log_dir + "/model")
    _, _ = env.reset()
    for i in tqdm(range(args.evaluation_steps)):
        action, _ = DRL_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("\n[Evaluation Step %d]: " % i)
        print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        print_info(info)
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
