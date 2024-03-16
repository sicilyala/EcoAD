import numpy as np
from torch import nn
from stable_baselines3 import DQN, DDPG, TD3, SAC, A2C, PPO
from stable_baselines3.common.noise import NormalActionNoise

from common.my_utils import triangular_schedule, triangular2_schedule
from common.CnnNetwork import CustomCNN


def DQN_agent(env, args, log_dir, action_dim):
    DRL_agent = DQN(
        policy="MlpPolicy",
        env=env,
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
        tensorboard_log=log_dir,
    ) 
    return DRL_agent


def DDPG_agent(env, args, log_dir, action_dim):
    DRL_agent = DDPG(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=args.net_arch,  # default [256, 128]    args.net_arch
        ),
        learning_rate=triangular2_schedule(max_LR=args.LR, min_LR=args.LR_min),
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=NormalActionNoise(
            np.zeros(action_dim),  # config["action"]["action_dim"]
            np.zeros(action_dim) + args.noise,
        ),
        # TODO action-noise std, how to degenerate? seems impossible?
        # TODO how eliminate action noise when replaying ?
        # off_policy_algorithm: L398
        # replay_buffer_class='HerReplayBuffer',   # only Hindsight ER
        # replay_buffer_kwargs=env,        #
        verbose=2,  # info output
        seed=args.seed,
        device=args.device,
        tensorboard_log=log_dir,
        # _init_setup_model=False,
        # actor_activation_fn_ELU=True,
    ) 
    return DRL_agent


def TD3_agent(env, args, log_dir, action_dim):
    pass


def SAC_agent(env, args, log_dir, action_dim):
    # print("\nSAC AGENT is initialized here.\n")
    DRL_agent = SAC(
        policy="CnnPolicy",
        # policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=args.features_dim),
            net_arch=args.net_arch,  # default [256, 128]    args.net_arch
        ),
        learning_rate=triangular2_schedule(max_LR=args.LR, min_LR=args.LR_min),  
        # learning_rate=triangular_schedule(max_LR=args.LR, min_LR=args.LR_min),
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef="auto_1",  # 'auto_0.1' for using 0.1 as initial value
        # TODO hwo to use determined entropy coefficient ? (how to eliminate actin noise ?)
        target_entropy=-action_dim,
        # replay_buffer_class='HerReplayBuffer',   # only Hindsight EP
        # replay_buffer_kwargs=env,
        verbose=2,
        seed=args.seed,
        device=args.device,
        tensorboard_log=log_dir,
        # _init_setup_model=False,
    )    
    return DRL_agent


def A2C_agent(env, args, log_dir, action_dim):
    pass


def PPO_agent(env, args, log_dir, action_dim):
    pass


DRL_methods = {"dqn": DQN, "ddpg": DDPG, "td3": TD3, "sac": SAC, "a2c": A2C, "ppo": PPO}

DRL_agents = {
    "dqn": DQN_agent,
    "ddpg": DDPG_agent,
    "td3": TD3_agent,
    "sac": SAC_agent,
    "a2c": A2C_agent,
    "ppo": PPO_agent,
}
