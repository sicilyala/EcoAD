import os
import sys
import time
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from torchsummary import summary

from highway_env import register_highway_envs
from common.arguments import get_args
from common.env_config import show_config, get_config
from common.my_utils import print_info, print_obs
from common.drl_agents import DRL_methods, DRL_agents


if __name__ == "__main__":
    print("[platform]: %s" % sys.platform)
    args = get_args()
    config = get_config(args)
    # register new environments
    register_highway_envs()
    # env = gym.make('highway-v0', render_mode='rgb_array')
    env = gym.make("EcoAD-v0", render_mode="rgb_array", config=config)
    # env = gym.make('EcoAD-v0', render_mode='rgb_array')    # can't pass [config] to [env] immediately # 第一次定义无法将config传入env env.env.configure(config)
    obs, info = env.reset()
    print("\n----------Reset Before Training----------")
    print_obs(
        obs,
        ems_flag=config["action"]["ems_flag"],
        obs_features=config["observation"]["features"],
    )
    print_info(info)
    show_config(config)
    print("observation_space: ", env.observation_space)
    print("action_space: ", env.action_space)
    print("observation_shape: ", env.observation_space.shape)
    print("action_shape: ", env.action_space.shape)
    print("----------------------------------\n")
    print(
        "\n----------Training started at %s----------"
        % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    )

    # dir
    log_dir = "./EcoHighway_DRL/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if args.lateral_control:
        args.log_dir += "_Lateral"
    if args.ems_flag:
        args.log_dir += "_EMS"
    log_dir += args.log_dir
    model_name = args.drl_model.lower()

    # DRL agent training
    DRL_agent = DRL_agents[model_name](
        env, args, log_dir, action_dim=config["action"]["action_dim"]
    )
    print(DRL_agent.policy)
    summary(
        model=DRL_agent.policy, input_size=(1, obs.shape[0], obs.shape[1])
    )  # C*H*W, the same as input
    DRL_agent.learn(total_timesteps=args.total_time_steps, log_interval=1)
    now = time.localtime()
    now_str = time.strftime("%b-%d-%H-%M", now)
    DRL_agent.save(log_dir + "/%s-model-%s" % (model_name, now_str))
    del DRL_agent

    # evaluation: Load and test the saved model
    DRL_agent = DRL_methods[model_name].load(
        log_dir + "/%s-model-%s" % (model_name, now_str)
    )
    print(
        "\n----------Training stopped at %s----------"
        % time.strftime("%Y-%m-%d %H:%M:%S", now)
    )
    print("\n----------Start Evaluating----------")
    obs, info = env.reset()
    for i in tqdm(range(args.evaluation_episodes)):
        action, _ = DRL_agent.predict(obs[np.newaxis, :], deterministic=True)
        # print('action: ', action, type(action), action.shape)
        obs, reward, terminated, truncated, info = env.step(action[0])
        # print("\n[Evaluation Step %d]: " % i)
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
