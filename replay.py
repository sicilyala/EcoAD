import sys
import os
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io as scio

from highway_env import register_highway_envs
from common.arguments import get_args
from common.env_config import get_config
from common.drl_agents import DRL_methods
from common.my_utils import print_obs, print_info


if __name__ == "__main__":
    # python .\replay.py --drl_model 'ddpg' --model_time Jan-17-11-35

    print("Command line arguments: ", sys.argv)
    # env config
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    log_dir = "./EcoHighway_DRL/" + args.dir_name + "/"

    # replay the video
    print("\n----------Start Evaluating----------")
    model_name = args.drl_model.lower()
    model_dir = log_dir + model_name + "-model-%s" % args.model_time
    data_dir = model_dir + "-data"
    DRL_agent = DRL_methods[model_name].load(model_dir)    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    env.configure({"simulation_frequency": 15})
    obs, info = env.reset() 
    for i in tqdm(range(args.evaluation_episodes)):
        action, _ = DRL_agent.predict(obs[None], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        scio.savemat(data_dir+"/step%d.mat" % i, mdict=info) 
        # print("\n[Evaluation Step %d]: " % i)
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    
    plt.imshow(env.render())
    # plt.show()
