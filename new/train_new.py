import numpy as np
import os
import shutil
import torch
import matplotlib.pyplot as plt

from argument import get_args
from crowd_nav.configs.config import Config
from rl.networks.envs import make_vec_envs

def main():
    algo_args = get_args()

    if not os.path.exists(algo_args.output_dir):
        os.makedirs(algo_args.output_dir)
    elif not algo_args.overwrite:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite to overwrite.".format(algo_args.output_dir))
    
    save_config_dir = os.path.join(algo_args.output_dir, 'configs')
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy('crowd_nav/configs.config.py', save_config_dir)
    shutil.copy('crowd_nav/configs/__init__.py', save_config_dir)
    shutil.copy('arguments.py', algo_args.output_dir)

    env_config = Config()
    config = Config()
    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")

    env_name = algo_args.env_name

    if config.sim.render:
        algo_args.num_processes = 1
        algo_args.num_mini_batch = 1

    if config.sim.render:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.ion()
        plt.show()
    else:
        ax = None

    envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes, algo_args.gamma, None, device, False, env_config, ax=ax, pretext_wrapper=config.env.use_wrapper)
    