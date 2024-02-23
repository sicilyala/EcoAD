from typing import Dict, Text
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class EcoADEnv(AbstractEnv):
    name = 'EcoADEnv'
    """
    An ecological autonomous driving (ECO-AD) environment in highway scenario.

    The vehicle is driving on a straight highway with several lanes
    """

    @classmethod
    def default_config(cls) -> dict:
        # print("in ecoad_env.py default_config, L24")
        # print(super())
        # print(EcoADEnv.mro())
        # print('\n***** main.py is in: %s *****' % os.getcwd())
        # print('\n***** The env is in: highway_env.envs:EcoADEnv *****\n')

        de_config = super().default_config()  # a dict
        de_config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return de_config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.vehicle.AgentEMS.reset_obs()
        # print("EcoADEnv: _reset(): AgentEMS.reset_obs()")

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(lanes=self.config["lanes_count"],
                                                                   start=self.config["lane_start"],
                                                                   length=self.config["lane_length"],
                                                                   speed_limit=self.config["road_spd_limit"]), 
                                                                   np_random=self.np_random, 
                                                                   record_history=self.config["show_trajectories"])
        self.lanes_list = self.road.network.lanes_list()
        self.lanes_centers = {}     # 
        for lane in self.lanes_list:            
            lane_width = lane.width_at(lane.start)
            # print(lane_width)
            lane_id = self.road.network.get_closest_lane_index(lane.start)[2] 
            lane_center = (0.5+lane_id)*lane_width
            self.lanes_centers.update({lane_id: lane_center}) 
            # print('lane {}, center at {}, start at {}, end at {}'.format(lane_id, lane_center, lane.start, lane.end))
        # print(self.lanes_centers)
        # print(self.lanes_list)  
        self.lane_width = lane_width

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # "highway_env.vehicle.behavior.IDMVehicle"
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # already weighted average rewards
        total_rewards = self._rewards(action)
        total_reward_weighted_sum = sum(reward for reward in total_rewards.values()) 
        
        if self.config["normalize_reward"]:
            print("EcoADEnv._reward(): 'normalize_reward' yes")
            total_reward_weighted_sum = utils.lmap(total_reward_weighted_sum,
                                                    [self.config["collision_reward"], 
                                                     self.config["high_speed_reward"] + self.config["right_lane_reward"]], 
                                                     [0, 1])
             
        return total_reward_weighted_sum

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)  # return: all lanes belonging to the same road.
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        
        # safety reward 
        r_safety = -2.0 if self.vehicle.crashed else 1.0  
        
        # on road reward
        r_road  = 1.0 if self.vehicle.on_road else -2.0     # TODO 与中心线合并，与最左车道合并?
        
        # leftmost lane reward 
        r_left = lane / max(len(neighbours) - 1, 1)    # len(neighbours): number os lanes, range: [0, 1, 2]/2
        
        # driving on the center line of lane
        lane_center_lateral_position = self.lanes_centers[lane]
        vehicle_lateral_position = self.vehicle.position[1]    # y coordinate 
        r_center = 0.5*self.lane_width - abs(vehicle_lateral_position - lane_center_lateral_position) # maximum: 2
        
        # high speed reward, efficiency 
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        r_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])     # 20 -> -1.0, 30 -> 1.0 
        
        # comfort reward 
        r_jerk = 2 - abs(action[1] * forward_speed) / 4    # action[1]: [maximum 0.7854]; r_jerk maximum: 2
        # TODO r_jerk is it suitable? waiting for further study     
        
        # TODO 是否需要单独的换道奖励？添加在速度奖励后面？
              
        rewards = {
            "collision_reward": float(r_safety), 
            "on_road_reward": float(r_road),            
            "left_lane_reward": float(r_left),
            "center_line_reward": float(r_center),
            "high_speed_reward": float(r_speed),
            "comfort_reward": float(r_jerk)
        }           
        
        # EMS reward              
        if self.config["action"]["ems_flag"]:                   
            r_EMS = self.vehicle.EMS_reward
            rewards.update({"EMS_reward": float(r_EMS)})

        # weighted average rewards 
        weighted_rewards = {}
        for name, reward in rewards.items():
            w_reward = reward * self.config.get(name, 0)
            weighted_rewards.update({name: w_reward})
        
        return weighted_rewards

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        # TODO 为了延长episode，需要减少off-road次数，考虑连续两次off-road才terminate？
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached.""" 
        return self.time >= self.config["duration"]
