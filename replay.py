import sys
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

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
    DRL_agent = DRL_methods[model_name].load(
        log_dir + model_name + "-model-%s" % args.model_time
    )

    obs, info = env.reset()
    for i in tqdm(range(args.evaluation_episodes)):
        action, _ = DRL_agent.predict(obs[None], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action[0])
        # print("\n[Evaluation Step %d]: " % i)
        # print_obs(obs, ems_flag=config["action"]["ems_flag"], obs_features=config["observation"]["features"])
        # print_info(info)
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
