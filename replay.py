import sys
import os
import gymnasium as gym
from tqdm import trange
from matplotlib import pyplot as plt
import scipy.io as scio

from highway_env import register_highway_envs
from common.arguments import get_args
from common.env_config import get_config
from common.drl_agents import DRL_methods
from common.my_utils import print_obs, print_info


def replay(env, drl_agent, replay_steps, model_dir):
    data_dir = model_dir + "-data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    reset_step = []
    # env.configure({"simulation_frequency": 30})
    obs, _ = env.reset() 
    for i in trange(replay_steps, desc='replaying', unit='step'):
        action, _ = drl_agent.predict(obs[None], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        scio.savemat(data_dir+"/step%d.mat" % i, mdict=info) 
        
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            last_reset = reset_step[-1] if reset_step else 0
            print("[reset] at step %d, safe driving lasts for %d steps." % (i, i-last_reset))
            reset_step.append(i)
            obs, _ = env.reset()
    print('reset steps: ', reset_step)
    scio.savemat(data_dir+"/reset_step.mat", mdict={'reset_step': reset_step}) 
    plt.imshow(env.render())
    # plt.show()
    

if __name__ == "__main__":
    # python .\replay.py --drl_model 'ddpg' --model_time Jan-17-11-35

    print("Command line arguments: ", sys.argv)
    # env config
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    log_dir = "./EcoHighway_DRL/" + args.dir_name + "/"    
    model_name = args.drl_model.lower()
    model_dir = log_dir + model_name + "-model-%s" % args.model_time
    DRL_agent = DRL_methods[model_name].load(model_dir)    

    # replay the video
    print("\n----------Start Evaluating----------")
    replay(env, DRL_agent, args.replay_steps, model_dir)
    