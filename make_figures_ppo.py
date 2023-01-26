import os
import sys
import json
from glob import glob

import pandas as pd

import multiprocessing as mp


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

LIB_DIR = os.path.join(CURR_DIR, 'libs')
sys.path.append(LIB_DIR)
from experiment_utils import load_experiment

from gym_utils import load_robot
from figure_drawer import EvogymControllerDrawerPPO, pool_init_func
import custom_envs.parkour

from arguments.evogym_ppo import get_figure_args


def main():

    args = get_figure_args()

    poet_path = os.path.join(CURR_DIR, 'out', 'evogym_poet', args.name)
    poet_args = load_experiment(poet_path)

    niche_path = os.path.join(poet_path, 'niche', str(args.key), )
    terrain_file = os.path.join(niche_path, 'terrain.json')

    expt_path = os.path.join(niche_path, 'ppo_result')
    expt_args = load_experiment(expt_path)



    robot = load_robot(CURR_DIR, poet_args['robot'])


    controller_files = {}
    for trial_dire in glob(expt_path + '/*/'):
        trial_num = os.path.basename(trial_dire[:-1])
        if not trial_num.isdigit():
            continue
        history_file = os.path.join(trial_dire, 'history.csv')
        history = pd.read_csv(history_file)
        best_iteration = history.loc[history['reward'].idxmax(), 'iteration']
        controller_file = os.path.join(trial_dire, 'controller', f'{best_iteration}.pt')
        controller_files[trial_num] = controller_file

    figure_path = os.path.join(expt_path, 'figure')
    draw_kwargs = {}
    if args.save_type=='gif':
        draw_kwargs = {
            'resolution': (1280*args.resolution_ratio, 720*args.resolution_ratio),
            'deterministic': True
        }
    elif args.save_type=='jpg':
        draw_kwargs = {
            'interval': args.interval,
            'resolution_scale': args.resolution_scale,
            'timestep_interval': args.timestep_interval,
            'distance_interval': args.distance_interval,
            'display_timestep': args.display_timestep,
            'deterministic': True
        }
    drawer = EvogymControllerDrawerPPO(
        save_path=figure_path,
        env_id=poet_args['task'],
        robot=robot,
        overwrite=not args.not_overwrite,
        save_type=args.save_type, **draw_kwargs)

    draw_function = drawer.draw


    if not args.no_multi:

        lock = mp.Lock()
        pool = mp.Pool(args.num_cores, initializer=pool_init_func, initargs=(lock,))
        jobs = []

        for trial_num,controller_file in controller_files.items():
            jobs.append(pool.apply_async(draw_function, args=(trial_num, terrain_file, controller_file)))

        for job in jobs:
            job.get(timeout=None)


    else:

        lock = mp.Lock()
        lock = pool_init_func(lock)

        for trial_num,controller_file in controller_files.items():
            draw_function(trial_num, terrain_file, controller_file)

if __name__=='__main__':
    main()
