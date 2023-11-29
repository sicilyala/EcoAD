import os
import numpy as np
import pandas as pd
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from highway_env import register_highway_envs
from arguments import get_args
from env_config import show_config, get_config


# TEST FOR PYCHARM AND GITHUB

def print_obs(obs_table):
    rows = len(obs_table)
    index = ["car-0"]
    for i in range(1, rows - 1):
        index.append("car-%d" % i)
    if config["action"]["ems_flag"]:
        index.append("ems-obs")
    else:
        index.append("car-%d" % (rows - 1))
    obs_pd = pd.DataFrame(obs_table, index=index, columns=config["observation"]["features"])
    sn = 62
    print("\nObservation Table")
    print("*" * sn)
    print(obs_pd)
    print("*" * sn)


def print_info(info_dict):
    sn = 60
    print("\nInformation Table")
    print("*" * sn)
    for key, value in info_dict.items():
        if type(value) is dict:
            print("\n*****%s sub-table*****" % key)
            for key1, value1 in value.items():
                print("{}, {}".format(key1, value1))
        else:
            print("{}, {}".format(key, value))
    print("*" * sn)


if __name__ == '__main__':
    args = get_args()
    config = get_config(args)
    # register new environments
    register_highway_envs()
    # env = gym.make('highway-v0', render_mode='rgb_array')
    env = gym.make('EcoAD-v0', render_mode='rgb_array', config=config)
    # env = gym.make('EcoAD-v0', render_mode='rgb_array')    # can't pass [config] to [env] immediately
    # 第一次定义无法将config传入env env.env.configure(config)
    obs, info = env.reset()
    print("\n----------Training Stage----------")
    print_obs(obs)
    print_info(info)
    show_config(config)
    print('observation_space: ', env.observation_space)
    print('action_space: ', env.action_space)
    print('----------------------------------\n')

    # DRL agent learning
    log_dir = "./EcoHighway_DRL/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.ems_flag:
        args.log_dir += "_EMS"
    log_dir += args.log_dir
    if config["ActionContinuity"]:
        DRL_agent = DDPG(policy='MlpPolicy', env=env,
                         policy_kwargs=dict(net_arch=args.net_arch),
                         learning_rate=args.LR,
                         buffer_size=args.buffer_size,
                         learning_starts=args.learning_starts,
                         batch_size=args.batch_size,
                         tau=args.tau,
                         gamma=args.gamma,
                         train_freq=args.train_freq,
                         gradient_steps=args.gradient_steps,
                         action_noise=NormalActionNoise(np.zeros(config["action"]["action_dim"]),
                                                        np.zeros(config["action"]["action_dim"]) + 0.5),
                         # replay_buffer_class='HerReplayBuffer',
                         # replay_buffer_kwargs=env,        # ?
                         verbose=2,  # info output
                         seed=args.seed,
                         device='auto',
                         # _init_setup_model=False,
                         tensorboard_log=log_dir)
        DRL_agent.learn(total_timesteps=args.total_time_steps)
        DRL_agent.save(log_dir + "/model")
        # remove to demonstrate saving and loading
        del DRL_agent
        # Load and test saved model
        DRL_agent = DDPG.load(log_dir + "/model")
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
                        device='auto',
                        tensorboard_log=log_dir)
        DRL_agent.learn(total_timesteps=args.total_time_steps)
        DRL_agent.save(log_dir + "/dqn_model")
        del DRL_agent
        DRL_agent = DQN.load(log_dir + "/dqn_model")

    # evaluation
    print("\n----------Validating Stage----------")
    _, _ = env.reset()
    for _ in range(args.evaluation_steps):
        action, _ = DRL_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print_obs(obs)
        print_info(info)
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
