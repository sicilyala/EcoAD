import numpy as np 


""" define configurations"""
def get_config(argus):
    ActionContinuity = argus.action_continuity
    LateralControl = argus.lateral_control
    MAX_SPD = argus.max_spd
    EMS_flag = argus.ems_flag

    configs = {
        "envname": 'cwqaq-ecoad',
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,  # Number of observed vehicles
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "ems_features": (['SOC', 'SOH_FCS', 'SOH_BAT', 'P_FCS', 'P_req'] if EMS_flag else []),
            # len(ems_features) must less than len(features)
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-MAX_SPD, MAX_SPD],
                "vy": [-MAX_SPD, MAX_SPD],
                "P_FCS": [0, 60],
                "P_req": [-200, 200],},
            "clip": True,  # Should the value be clipped in the desired range
            "absolute": False,
            "order": "sorted",
        },
        "ActionContinuity": ActionContinuity,
        "action": {
            "type": "ContinuousAction" if ActionContinuity else "DiscreteMetaAction",
            "acceleration_range": [-2.0, 2.0],  # m/s2
            "speed_range": [-MAX_SPD, MAX_SPD],  # m/s
            "lateral": LateralControl,
            "steering_range": [-np.pi / 4, np.pi / 4],  # rad
            "ems_flag": EMS_flag,
            "engine_power_range": [0, 60],  # kW
            # "dynamical": False,      # # False for Vehicle, True for BicycleVehicle (with tire friction and slipping)
            "action_dim": int(1 + LateralControl + EMS_flag)},
        # reward
        "normalize_reward": False,
        "reward_speed_range": [MAX_SPD-5, MAX_SPD],
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "on_road_reward": 1.0,  # True of False
        "offroad_terminal": True, # activate off-road terminal 
        "lane_change_reward": 1.0,  # The reward received at each lane change action.
        "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
        "high_speed_reward": 1.0,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
        "EMS_reward": 1.0,     # it's actually the weight coefficient
        # environment
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": 3,
        "vehicles_density": 1.5,
        "vehicles_count": 500,
        "initial_spacing": 3,
        "duration": 100,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "screen_width": 600,  # [px]
        "screen_height": 350,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }
    return configs


def show_config(configs):
    print('\n----------------------------------')
    print('env-name: ', configs["envname"])
    print('observation_type: ', configs["observation"]["type"])
    print('action_type: ', configs["action"]["type"])
    print('lanes_count: ', configs["lanes_count"])
    print('----------------------------------')


if __name__ == '__main__':
    import gymnasium as gym
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from stable_baselines3 import DQN, DDPG
    from highway_env import register_highway_envs
    from arguments import get_args
    from env_config import show_config, get_config


    args = get_args()
    args.action_continuity = True
    args.lateral_control = False
    args.ems_flag = True
    config = get_config(args) 
    config["lanes_count"] = 3 
    config["vehicles_density"] = 2 
    config["vehicles_count"] = 500  

    register_highway_envs() 
    env = gym.make('EcoAD-v0', render_mode='rgb_array', config=config)

    obs, info = env.reset() 
    for i in tqdm(range(100)):
        # action, _ = DRL_agent.predict(obs, deterministic=True)
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, terminated, truncated, info = env.step(action) 
        env.render()
        if terminated or truncated:
            _, _ = env.reset()

    plt.imshow(env.render())
    plt.show()
