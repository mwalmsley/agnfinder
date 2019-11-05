import os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from agnfinder.tf_sampling import run_sampler


def check_parameter_bias(samples, true_params):
    best_guess = np.median(samples, axis=1)
    fig, axes = plt.subplots(nrows=7, figsize=(8, 4 * 6))
    for param_n, _ in enumerate(range(7)):  # TODO
        ax = axes[param_n]
        ax.scatter(true_params[:, param_n], best_guess[:, param_n])
        ax.set_xlabel('True {}'.format(param_n))
        ax.set_ylabel('Best guess')
    fig.tight_layout()
    return fig, axes


if __name__ == '__main__':

    """
    Check if the emulated HMC sampling is correctly recovering the original galaxy parameters for the forward model.

    Example use:
    /data/miniconda3/envs/agnfinder/bin/python /Data/repos/agnfinder/agnfinder/tf_sampling/evaluate_performance.py --save-dir results/emulated_sampling/latest_1_32000_random
    """

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    args = parser.parse_args()
    save_dir = args.save_dir

    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    logging.info('Loading samples')
    samples, true_params, _ = run_sampler.read_performance(save_dir)

    logging.info('Checking parameter bias')
    fig, _ = check_parameter_bias(samples, true_params)
    fig.savefig(os.path.join(save_dir, 'parameter_bias.pdf'))
