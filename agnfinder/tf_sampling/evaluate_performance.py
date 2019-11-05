import os
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

    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    # save_dir = 'results/recovery/latest_roughly_correct'
    save_dir = 'results/recovery/latest_random'

    logging.info('Loading samples')
    samples, true_params, true_observations = run_sampler.read_performance(save_dir)

    logging.info('Checking parameter bias')
    fig, axes = check_parameter_bias(samples, true_params)
    fig.savefig(os.path.join(save_dir, 'parameter_bias.pdf'))
