import argparse


def get_args():
    parser = argparse.ArgumentParser(description='environment and algorithm')
    # environment configuration
    parser.add_argument('--action_continuity', default=True, type=lambda x: x.lower() == 'true', help='Action type')
    parser.add_argument('--lateral_control', default=False, type=lambda x: x.lower() == 'true', help='activate lateral action')
    parser.add_argument('--ems_flag', default=True, type=lambda x: x.lower() == 'true', help='activate EMS,True of False')
    parser.add_argument('--max_spd', default=25, type=float)

    # DRL method parameters
    parser.add_argument('--net_arch', default=[256, 256], type=list, help='policy net arch')
    parser.add_argument('--LR', default=1e-3, type=float, help='learning_rate')
    parser.add_argument('--buffer_size', default=100000, type=int, help='buffer_size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--tau', default=0.005, type=float, help='tau')
    parser.add_argument('--gamma', default=0.995, type=float, help='discount rate')
    parser.add_argument('--device', default='auto', type=str, help="auto, cuda, cpu")
    parser.add_argument('--total_time_steps', default=20000, type=int,
                        help="the total number of samples (env steps) to train on")
    parser.add_argument('--learning_starts', default=1000, type=int,
                        help='how many steps for the DRL agent to collect transitions before starting learning ')
    parser.add_argument('--train_freq', default=1, type=int, help='train_freq')
    parser.add_argument('--gradient_steps', default=1, type=int, help='gradient_steps')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--log_dir', default="test", type=str, help='log_dir')
    parser.add_argument('--evaluation_steps', default=100, type=int,
                        help="the total number of env steps evaluate")

    args = parser.parse_args()
    return args
