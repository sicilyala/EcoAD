# calculate the traffic density of the road
import numpy as np


def traffic_density(vehicles_density, speed=20):
    lane_number = 3
    # speed = 20      # random from [20, 25] m/s
    spacing = 1 / vehicles_density
    default_spacing = 12+1.0*speed
    offset = spacing * default_spacing * np.exp(-5 / 40 * lane_number)  # vehicle distance
    density = 1 + 1000 / (offset * 1.0)
    return density

vehicles_density = [1, 1.5, 2, 2.25, 2.5, 3]
traffic_density_list = {}
for density in vehicles_density:
    traffic_density_list[density] = traffic_density(density) 
print(traffic_density_list)

"""
vehicle_density = [1, 1.5, 2, 2.5, 2.25, 3]
vehicle distance: m
{1: 21.99325692131111, 1.5: 14.662171280874073, 2: 10.996628460655556, 2.5: 8.797302768524444, 3: 7.331085640437037} 
traffic density: v/km/ln
{1: 46.46848170681879, 1.5: 69.2027225602282, 2: 91.93696341363759, 2.5: 114.67120426704697, 3: 137.4054451204564}
{1: 46.46848170681879, 1.5: 69.2027225602282, 2: 91.93696341363759, 
    2.25: 103.30408384034229, 2.5: 114.67120426704697, 3: 137.4054451204564}
"""
