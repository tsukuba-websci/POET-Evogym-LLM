
import sys
import os
import json

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)
from experiment_utils import initialize_experiment, load_experiment


from gym_utils import load_robot
import custom_envs.parkour

from run_ppo import run_ppo

from arguments.evogym_ppo import get_args

class ppoConfig:
    def __init__(self, args):
        self.num_processes = args.num_processes
        self.eval_processes = 1
        self.seed = 1

        self.steps = args.steps
        self.num_mini_batch = args.num_mini_batch
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma
        self.clip_range = args.clip_range
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr_decay = True
        self.gae_lambda = 0.95
        self.init_log_std = args.init_log_std


def main():

    args = get_args()

    expt_path = os.path.join(CURR_DIR, 'out', 'evogym_poet', args.name)
    expt_args = load_experiment(expt_path)

    niche_path = os.path.join(expt_path, 'niche', str(args.key))
    assert os.path.exists(niche_path), f'no niche key {args.key}'

    result_path = os.path.join(niche_path, 'ppo_result')
    initialize_experiment(args.name, result_path, args)

    robot = load_robot(CURR_DIR, expt_args['robot'])
    terrain_file = os.path.join(niche_path, 'terrain.json')
    terrain = json.load(open(terrain_file, 'r'))

    env_kwargs = dict(**robot, terrain=terrain)

    ppo_config = ppoConfig(args)

    for i in range(args.num):
        print(f'----------start ppo learning {i+1: 2}----------')
        save_path = os.path.join(result_path, str(i+1))
        controller_path = os.path.join(save_path, 'controller')
        os.makedirs(controller_path, exist_ok=True)

        history_file = os.path.join(save_path, 'history.csv')
        run_ppo(
            env_id=expt_args['task'],
            robot=env_kwargs,
            train_iters=args.train_iters,
            eval_interval=args.evaluation_interval,
            save_file=controller_path,
            config=ppo_config,
            deterministic=True,
            save_iter=True,
            history_file=history_file)
        print()


if __name__=='__main__':
    main()