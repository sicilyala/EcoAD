import sys  
import gymnasium as gym
from stable_baselines3 import DQN, DDPG, TD3, SAC, A2C, PPO 
from tqdm import tqdm
from matplotlib import pyplot as plt

from highway_env import register_highway_envs
from arguments import get_args
from env_config import get_config
from utils import print_obs, print_info


if __name__=='__main__': 
    # python evaluate.py --model_name sac --model_time 2023-12-23-22:55   
    # python evaluate.py --dir_name test_EMS --lateral_control False --model_name ddpg --model_time 2023-12-19-11:43
    print("Command line arguments: ", sys.argv) 
    # env config
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make('EcoAD-v0', render_mode='rgb_array', config=config)  
    log_dir = "./EcoHighway_DRL/" + args.dir_name + "/"
    # replay the vedio
    print("\n----------Start Evaluating----------") 

    name_agent = {'dqn': DQN, 'ddpg': DDPG, 'td3': TD3, 
                  'sac': SAC, 'a2c': A2C, 'ppo': PPO} 
    model_name = args.model_name.lower() 
    DRL_agent = name_agent[model_name].load(log_dir + model_name + "-model-%s" % args.model_time)
 
    obs, info = env.reset() 
    for i in tqdm(range(args.evaluation_episodes)):
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
