import os
import sys
import time 
import gymnasium as gym 
from torchsummary import summary 

from highway_env import register_highway_envs
from common.arguments import get_args
from common.env_config import show_config, get_config
from common.my_utils import print_info, print_obs, print_Hr_Min, extract_len_rew
from common.drl_agents import DRL_agents 
from common.AutoLog import AutoLog
from replay import replay


if __name__ == "__main__":
    args = get_args()
    config = get_config(args)
    
    # register new environments
    register_highway_envs()
    # env = gym.make('highway-v0', render_mode='rgb_array')
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    # env = gym.make('EcoAD-v0', render_mode='rgb_array')    # can't pass [config] to [env] immediately # 第一次定义无法将config传入env env.env.configure(config)
    obs, info = env.reset()

    # dir
    log_dir = "./EcoHighway_DRL/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.lateral_control:
        args.log_dir += "_Lateral"
    if args.ems_flag:
        args.log_dir += "_EMS"
    log_dir += args.log_dir
    drl_model = args.drl_model.lower()

    # define DRL agent 
    DRL_agent = DRL_agents[drl_model](env, args, log_dir, action_dim=config["action"]["action_dim"])
    
    # print logger   
    log_name = 'training_at_%s.log' % time.ctime()
    sys.stdout = AutoLog(filepath=log_dir, filename=log_name)
    print("\n[platform]: %s" % sys.platform)
    print("\nCommand line arguments: ", sys.argv)
    print("\n----------Reset Before Training----------")
    print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
    print_info(info)
    show_config(config)
    print("observation_space: ", env.observation_space)
    print("action_space: ", env.action_space)
    print("observation_shape: ", env.observation_space.shape)
    print("action_shape: ", env.action_space.shape)
    print("----------------------------------\n")
    print("\n----------Training started at %s----------"% time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("\n------------%s model structure------------" % drl_model.upper())
    print(DRL_agent.policy)
    summary(model=DRL_agent.policy, input_size=(1, obs.shape[0], obs.shape[1]))  # input_size C*H*W, the same as input 
    
    # training 
    st_time = time.time()
    DRL_agent.learn(total_timesteps=args.total_time_steps, log_interval=1)
    logger_dir = DRL_agent.logger.dir    
    logger_id = int(logger_dir.split('_')[-1]) 
    model_dir = logger_dir + "/learned_%s_model" % (drl_model.upper())      # saved in .zip
    DRL_agent.save(model_dir)
    
    # define saving directory  
    # now = time.localtime()    
    # now_str = time.strftime("%b-%d-%H-%M", now)
    # model_dir = log_dir + "/%s-%d-%s" % (drl_model, logger_id, now_str)
    # print("\n----------Training stopped at %s----------" % time.strftime("%Y-%m-%d %H:%M:%S", now))  

    print("\n----------Training stopped at %s----------" %  time.ctime()) 
    print_Hr_Min(time.time()-st_time)
    
    buffer_counter = DRL_agent.replay_buffer.counter
    buffer_size = args.buffer_size
    ration = buffer_counter / buffer_size
    print("\nbuffer counter: %d, buffer_size: %d, ratio: %.3f" % (buffer_counter, buffer_size, ration))
    
    del DRL_agent     
    # evaluation: Load and test the learned model  
    replay(env, drl_model=drl_model, logger_dir=logger_dir, 
           replay_steps=args.replay_steps, sim_freq=args.sim_freq, vehicles_density=args.vehicles_density) 
    # change log dir 
    os.rename(log_dir+"/"+log_name, os.path.join(logger_dir, 'training.log'))
    extract_len_rew(logger_dir, 'training.log')       
