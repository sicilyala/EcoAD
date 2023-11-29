
def get_config(argus):
    ActionContinuity = argus.action_continuity
    LateralControl = argus.lateral_control
    MAX_SPD = argus.max_spd
    EMS_flag = argus.ems_flag

    configs = {
        "envname": 'cwqaq-ecoad',
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 6,  # Number of observed vehicles
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
            "steering_range": None,
            "lateral": LateralControl,
            "ems_flag": EMS_flag,
            "engine_power_range": [0, 60],  # kW
            # "dynamical": False,      # # False for Vehicle, True for BicycleVehicle (with tire friction and slipping)
            "action_dim": int(1 + LateralControl + EMS_flag)},
        # reward
        "normalize_reward": False,
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "on_road_reward": 1.0,  # True of False
        "lane_change_reward": 0,  # The reward received at each lane change action.
        "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
        "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
        "reward_speed_range": [20, 30],
        "EMS_reward": 10.0,     # it's actually the weight coefficient
        # environment
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": 3,
        "vehicles_density": 1,
        "vehicles_count": 100,
        "initial_spacing": 2,
        "duration": 400,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
        "screen_width": 600,  # [px]
        "screen_height": 350,  # [px]
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": True,
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
    import sys
    from arguments import get_args
    sys.modules["gym"] = gym

    args = get_args()
    config = get_config(args)
    env = gym.make('highway-v0')

    env.configure(config)
    obs, info = env.reset()
    print(obs)
    # env.render()
