import math


""" define configurations"""
def get_config(argus):
    ActionContinuity = argus.action_continuity
    LateralControl = argus.lateral_control
    MAX_SPD = argus.max_spd
    EMS_flag = argus.ems_flag
    action_frequency = argus.act_freq

    configs = {
        "envname": 'cwqaq-ecoad',
        # observation 
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
        # action 
        "ActionContinuity": ActionContinuity,
        "action": {
            "type": "ContinuousAction" if ActionContinuity else "DiscreteMetaAction",
            "acceleration_range": [-2.0, 2.0],  # m/s2
            "speed_range": [-MAX_SPD, MAX_SPD],  # m/s
            "lateral": LateralControl,
            "steering_range": [-math.pi / 4, math.pi / 4],  # 0.7854 rad
            "ems_flag": EMS_flag,
            "engine_power_range": [0, 60],  # kW
            # "dynamical": False,      # # False for Vehicle, True for BicycleVehicle (with tire friction and slipping)
            "action_dim": int(1 + LateralControl + EMS_flag)},
        
        # reward setting 
        "normalize_reward": False,
        "reward_speed_range": [MAX_SPD-5, MAX_SPD], # [25, 30]
        "offroad_terminal": True, # activate off-road terminal 
        
        # reward weight coefficients 
        "collision_reward": 1.0,  # The reward received when colliding with a vehicle.
        "on_road_reward": 1.0,  # True of False
        "left_lane_reward": 1.0, # The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes.
        "center_line_reward": 1.0,
        "high_speed_reward": 2.0,  # The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"].
        "comfort_reward": 1.0, 
        "EMS_reward": 1.0,     # it's actually the weight coefficient        
        "lane_change_reward": 1.0,  # The reward received at each lane change action.         
        
        # environment
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "lanes_count": 3,
        "lane_start": 0,
        "lane_length": 1000000,
        "road_spd_limit": MAX_SPD,      # m/s 
        "vehicles_density": 1.5,
        "vehicles_count": 500,
        "initial_spacing": 3,
        "duration": 100,  # [s]
        "simulation_frequency": action_frequency,  # [Hz] for fast training 
        "policy_frequency": action_frequency,  # [Hz] the action is executed for 0.5s, try [5, 10] HZ 
        "screen_width": 600,  # [px]
        "screen_height": 350,  # [px]
        "centering_position": [0.3, 0.5],       # what's this?
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
    print('obs_vehicles_count: ', configs["vehicles_count"])
    print('policy_frequency: ', configs['policy_frequency'])
    print('simulation_frequency: ', configs['simulation_frequency'])
    print('----------------------------------')
