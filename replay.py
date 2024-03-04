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


def replay(env, 
           drl_model: str,
           logger_dir: str, 
           replay_steps: int = 500,
           sim_freq: int = 100,
           ) -> None:
    model_dir = logger_dir + "/learned_%s_model" % (drl_model.upper())   
    DRL_agent = DRL_methods[drl_model.lower()].load(model_dir)       
    # print("\n------------%s model structure------------" % drl_model.upper())
    # print(DRL_agent.policy) 
    
    data_dir = logger_dir + "/replay_data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    print("\n---------- Evaluate %s using %s ----------" % (drl_model.upper(), logger_dir[-6:]))
    reset_step = [] 
    env.configure({"simulation_frequency": sim_freq})  
    print("action frequency: %d" % env.config["policy_frequency"]) 
    print("simulation frequency: %d" % env.config["simulation_frequency"])
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
    print("\nCommand line arguments: ", sys.argv)
    # env config
    register_highway_envs()
    args = get_args()
    config = get_config(args)
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    log_dir = "./EcoHighway_DRL/" + args.dir_name + "/"    
    drl_model = args.drl_model
    logger_dir = log_dir + drl_model.upper() + "_" + args.model_id
    # replay the video/
    replay(env, drl_model=drl_model, logger_dir=logger_dir, 
           replay_steps=args.replay_steps, sim_freq=args.sim_freq) 
    