# ECO-AD: Eco-Driving Framework for Hybrid Electric Vehicles in Multi-Lane Scenarios by Using Deep Reinforcement Learning Methods.

## Overview

The original implementation of **Eco-Driving Framework for Hybrid Electric Vehicles in Multi-Lane Scenarios by Using Deep Reinforcement Learning Methods.**

## Abstract

The eco-driving strategy is crucial for hybrid electric vehicles to save energy and reduce emissions. Most studies focused on longitudinal car-following or lane-changing maneuvers, lacking the consideration of continuous lateral dynamics, leading to insufficient optimization of energy-saving. This paper proposes an integrated eco-driving framework for fuel cell hybrid electric vehicles in multi-lane highway scenarios, in which trajectory planning and energy management are synchronously optimized by unified continuous control variables: acceleration, steering angle, and engine power, so as to maximize vehicle energy economy in real traffic environments. The key features of spatial traffic information and vehicular power conditions are extracted and formulated as the decision-making input. Then, the Soft Actor-Critic algorithm is utilized to optimize the eco-driving framework due to its good ability to explore complex strategy spaces for multi-objective optimization tasks. Analyses of the co-optimization process for motion trajectory planning and energy management show that, the proposed eco-driving strategy achieves better transverse-longitudinal comfort and energy economy by sacrificing 14.07% of the average speed, which results in an 87.65% improvement in the State-of-Health performance of the power system, and a reduction in the hydrogen consumption and the driving cost by 86.17% and 89.58%, respectively.

## Thanks

1. **The simulation environment is based on [Highway-env](https://github.com/Farama-Foundation/HighwayEnv).**

2. **The reinforcement learning algorithms are based on [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).**

## Citation

```BibTex
 @article{Chen_Peng_Ma_He_Ren_Wang_2026,
title={Eco-driving framework for hybrid electric vehicles in multi-lane scenarios by using deep reinforcement learning methods},
volume={5},
ISSN={27731537},
DOI={10.1016/j.geits.2025.100309},
number={2},
journal={Green Energy and Intelligent Transportation},
author={Chen, Weiqi and Peng, Jiankun and Ma, Yuhan and He, Hongwen and Ren, Tinghui and Wang, Chunhai},
year={2026},
month=apr,
pages={100309},
language={en}
}
```
