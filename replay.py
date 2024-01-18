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
# from common.my_utils import print_obs, print_info


def replay(env, replay_steps, model_name, model_dir):    
    DRL_agent = DRL_methods[model_name].load(model_dir)      
    data_dir = model_dir + "-data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    print("\n---------- Evaluating %s model ----------" % model_name.upper())
    reset_step = [] 
    env.configure({"simulation_frequency": args.sim_freq})
    obs, _ = env.reset() 
    for i in trange(replay_steps, desc='replaying', unit='step'):
        action, _ = DRL_agent.predict(obs[None], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        scio.savemat(data_dir+"/step%d.mat" % i, mdict=info) 
        
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            last_reset = reset_step[-1] if reset_step else 0
            print("\n[reset] at step %d, safe driving lasts for %d steps." % (i, i-last_reset))
            reset_step.append(i)
            obs, _ = env.reset()
   
    epi_mean_length = replay_steps/(1+len(reset_step))
    print('\nepi_mean_length: {:.1f}, reset steps: {}\n'.format(epi_mean_length, reset_step))
    reset_data = {'reset_step': reset_step, 'epi_mean_length': epi_mean_length}
    scio.savemat(data_dir+"/reset_data.mat", mdict=reset_data)  
    plt.imshow(env.render()) 
    

if __name__ == "__main__":
    # python .\replay.py --drl_model 'ddpg' --model_time Jan-17-11-35
    print("\nCommand line arguments: ", sys.argv)
    # env config
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    log_dir = "./EcoHighway_DRL/" + args.dir_name + "/"    
    model_name = args.drl_model.lower()
    model_dir = log_dir + model_name + "-model-%s" % args.model_time
    # replay the video
    replay(env, args.replay_steps, model_name, model_dir)
    