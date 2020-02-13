import os
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_context('notebook')
import corner
from tqdm.auto import tqdm  # notebook friendly
import glob
import arviz as az

from agnfinder.prospector.main import load_catalog
from agnfinder.prospector import load_photometry
from agnfinder import simulation_samples, simulation_utils
from agnfinder.tf_sampling import parameter_recovery, percentile_limits


def get_hpd(x, ci=0.8):
    if len(x) == 0:
        return np.array([np.nan, np.nan])
    return az.hpd(x[~np.isnan(x)], credible_interval=ci)


if __name__ == '__main__':

    sns.set_context('notebook')
    sns.set(font_scale=1.3)

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    args = parser.parse_args()

    params, marginals, true_params, samples = parameter_recovery.load_samples(args.save_dir, min_acceptance=0.6, max_redshift=4.0)

    dummy_array = np.zeros(42)  # anything
    n_param_bins = 10
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=n_param_bins)
    bin_centers = param_bins[1:] + (param_bins[0:-1] - param_bins[1:]) / 2

    samples_by_truth = [[] for n in range(len(param_bins))]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    all_axes = [ax for row in axes for ax in row]

    for which_param in range(8):

        ax = all_axes[which_param]

        samples_by_truth = [[] for n in range(len(bin_centers))]

        for galaxy_n in range(len(samples)):
            true_param = true_params[galaxy_n, which_param]
            true_param_index = np.digitize(true_param, param_bins) - 1
        #     print(true_param_index)
            samples_by_truth[true_param_index].append(np.squeeze(samples[galaxy_n, :, which_param]))
        # samples_by_truth = np.array(samples_by_truth)

        for n in range(len(samples_by_truth)):
            samples_by_truth[n] = np.array(samples_by_truth[n]).flatten()

        bounds_by_truth = np.array([get_hpd(x) for x in samples_by_truth])
        medians = np.array([np.median(x) for x in samples_by_truth])

        # delta_bounds_by_truth = bounds_by_truth.copy().transpose()
        # delta_bounds_by_truth[1, :] = delta_bounds_by_truth[1, :] - medians
        # delta_bounds_by_truth[0, :] = medians - delta_bounds_by_truth[0, :]

        ax.fill_between(bin_centers, bounds_by_truth[:, 0], bounds_by_truth[:, 1], alpha=0.5)
        ax.plot(bin_centers, bin_centers, linestyle='--', color='k')
        ax.set_xlabel('Truth')
        ax.set_ylabel(r'80% credible interval')

    fig.tight_layout()
    fig.savefig('results/latest_contours.png')