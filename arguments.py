import argparse

 
def get_args():
    parser = argparse.ArgumentParser(description='environment and algorithm')
    # environment configuration
    parser.add_argument('--action_continuity', default=True, type=lambda x: x.lower() == 'true', help='Action type')
    parser.add_argument('--lateral_control', default=False, type=lambda x: x.lower() == 'true', help='activate lateral action')
    parser.add_argument('--ems_flag', default=True, type=lambda x: x.lower() == 'true', help='activate EMS,True of False')
    parser.add_argument('--max_spd', default=30, type=float)

    # DRL method parameters
    parser.add_argument('--net_arch', default=[256, 128, 64], type=list, help='policy net arch')
    parser.add_argument('--LR', default=1e-3, type=float, help='maximal learning_rate')
    parser.add_argument('--LR_min', default=1e-5, type=float, help='minimal learning_rate')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--tau', default=0.001, type=float, help='tau')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount rate')
    parser.add_argument('--device', default='auto', type=str, help="auto, cuda, cpu")
    parser.add_argument('--noise', default=0.15, type=float, help='std of Gaussian noise')

    parser.add_argument('--buffer_size', default=1000, type=int, help='buffer_size')
    parser.add_argument('--total_time_steps', default=5200, type=int,
                        help="the total number of samples (env steps) to train on")
    
    parser.add_argument('--learning_starts', default=200, type=int,
                        help='how many steps for the DRL agent to collect transitions before starting learning ')
    parser.add_argument('--train_freq', default=1, type=int, help='Update the model every ``train_freq`` steps')
    parser.add_argument('--gradient_steps', default=-1, type=int, help='Set to ``-1`` means to do as many gradient steps as steps during the rollout')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--log_dir', default="test", type=str, help='log_dir')
    
    # for evaluation
    parser.add_argument('--evaluation_steps', default=10, type=int,
                        help="the total number of env steps evaluate")
    parser.add_argument('--dir_name', default="test_EMS", type=str)
    parser.add_argument('--model_name', default="ddpg-model", type=str, help='dqn-model')
    parser.add_argument('--model_time', default='', type=str)

    args = parser.parse_args()
    return args
