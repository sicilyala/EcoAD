import os
import time
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise
from tqdm import tqdm

from highway_env import register_highway_envs
from arguments import get_args
from env_config import show_config, get_config
from utils import print_info, print_obs, triangular_schedule, triangular2_schedule
 

if __name__ == '__main__':
    args = get_args()
    config = get_config(args)
    
    register_highway_envs() 
    env = gym.make('EcoAD-v0', render_mode='rgb_array', config=config) 
    obs, info = env.reset()

    print("\n----------Reset Before Training----------")
    print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
    print_info(info)
    show_config(config)
    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space)
    print('----------------------------------\n')
    print("\n----------Training started at %s----------" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # DRL agent learning
    log_dir = "./EcoHighway_DRL/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.lateral_control:
        args.log_dir += "_Lateral"
    if args.ems_flag:
        args.log_dir += "_EMS"
    log_dir += args.log_dir

    try:
        config["ActionContinuity"] is False
    except TypeError:
        print("DQN action space should be discrete")
    else: 
        DRL_agent = DQN(policy='MlpPolicy', env=env,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=5e-4,
                        buffer_size=15000,
                        learning_starts=args.learning_starts,
                        # how many steps of the model to collect transitions for before learning starts
                        batch_size=args.batch_size,
                        gamma=0.95,
                        train_freq=args.train_freq,
                        gradient_steps=args.gradient_steps,
                        target_update_interval=5,
                        verbose=2,
                        device=args.device,
                        tensorboard_log=log_dir)
        DRL_agent.learn(total_timesteps=args.total_time_steps)
        now = time.localtime()         
        DRL_agent.save(log_dir + "/dqn-model-%s" % time.strftime("%Y-%m-%d-%H:%M", now))
        del DRL_agent
        DRL_agent = DQN.load(log_dir + "/dqn-model-%s" % time.strftime("%Y-%m-%d-%H:%M", now))
    
    # evaluation  
    print("\n----------Training stopped at %s----------" % time.strftime("%Y-%m-%d %H:%M:%S", now))  
    print("\n----------Start Evaluating----------")
    _, _ = env.reset()
    for i in tqdm(range(args.evaluation_episodes)):
        action, _ = DRL_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # print("\n[Evaluation Step %d]: " % i)
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
