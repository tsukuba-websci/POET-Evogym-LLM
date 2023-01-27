import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description='Evogym PPO experiment'
    )

    parser.add_argument(
        'name',
        type=str,
        help='name of POET experiment'
    )
    parser.add_argument(
        'key',
        type=int,
        help='niche key'
    )

    parser.add_argument(
        '-n', '--num',
        default=5, type=int,
        help='how many times to run PPO (default: 5)'
    )

    parser.add_argument(
        '-p', '--num-processes',
        default=4, type=int,
        help='how many training CPU processes to use (default: 4)'
    )
    parser.add_argument(
        '-s', '--steps',
        default=128, type=int,
        help='num steps to use in PPO (default: 128)'
    )
    parser.add_argument(
        '-b','--num-mini-batch',
        default=4, type=int,
        help='number of batches for ppo (default: 4)'
    )
    parser.add_argument(
        '-e', '--epochs',
        default=4, type=int,
        help='number of ppo epochs (default: 4)'
    )
    parser.add_argument(
        '-i', '--train-iters',
        default=2500, type=int,
        help='learning iterations of PPO (default: 2500)'
    )
    parser.add_argument(
        '-ei', '--evaluation-interval',
        default=25, type=int,
        help='frequency of evaluation policy (default: 25)'
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        default=2.5e-4, type=float,
        help='learning rate (default: 2.5e-4)'
    )
    parser.add_argument(
        '--gamma',
        default=0.99, type=float,
        help='discount factor for rewards (default: 0.99)'
    )
    parser.add_argument(
        '-c', '--clip-range',
        default=0.1, type=float,
        help='ppo clip parameter (default: 0.1)'
    )
    parser.add_argument(
        '-std', '--init-log-std',
        default=0.0, type=float,
        help='initial log std of action distribution (default: 0.0)'
    )
    args = parser.parse_args()

    return args


def get_figure_args():
    parser = argparse.ArgumentParser(
        description='make robot figures'
    )

    parser.add_argument(
        'name',
        type=str,
        help='name of POET experiment'
    )
    parser.add_argument(
        'key',
        type=int,
        help='niche key'
    )

    parser.add_argument(
        '-st', '--save-type',
        type=str, default='gif',
        help='file type (default: gif, choose from [gif, jpg])'
    )

    parser.add_argument(
        '-tr', '--track-robot',
        action='store_true', default=False,
        help='track robot with camera in gif'
    )

    parser.add_argument(
        '-i', '--interval',
        type=str, default='timestep',
        help='in case of save type is jpg, type of interval for robot drawing (default: timestep, choose from [timestep, distance])'
    )
    parser.add_argument(
        '-rs', '--resolution-scale',
        type=float, default=32.0,
        help='resolution scale. <br> when output monochrome image, try this argument change. (default: 32.0)'
    )
    parser.add_argument(
        '--start-timestep',
        type=int, default=0,
        help='start timestep of render (default: 0, 0 means no blur)'
    )
    parser.add_argument(
        '-ti', '--timestep-interval',
        type=int, default=80,
        help='timestep interval for robot drawing (default: 80, if interval is hybrid, it should be about 40)'
    )
    parser.add_argument(
        '-b', '--blur',
        type=int, default=0,
        help='in case of jpg, timesteps for rendering motion blur (default: 0)'
    )
    parser.add_argument(
        '-bt', '--blur-temperature',
        type=float, default=0.6,
        help='blur temperature (default: 0.6, up to 1.0)'
    )
    parser.add_argument(
        '-di', '--distance-interval',
        type=float, default=0.8,
        help='distance interval for robot drawing'
    )
    parser.add_argument(
        '--display-timestep',
        action='store_true', default=False,
        help='display timestep above robot'
    )
    parser.add_argument(
        '--draw-trajectory',
        action='store_true', default=False,
        help='draw robot trajectory as line'
    )

    parser.add_argument(
        '-c', '--num-cores',
        default=1, type=int,
        help='number of parallel making processes (default: 1)'
    )
    parser.add_argument(
        '--not-overwrite',
        action='store_true', default=False,
        help='skip process if already figure exists (default: False)'
    )
    parser.add_argument(
        '--no-multi',
        action='store_true', default=False,
        help='do without using multiprocessing. if error occur, try this option. (default: False)'
    )

    args = parser.parse_args()

    assert args.name is not None, 'argumented error: input "{experiment name}"'

    return args
