import os
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import arviz
import corner
from tqdm import tqdm
import seaborn as sns

from agnfinder.tf_sampling import run_sampler

def get_valid_galaxy_indices(all_samples):

    logging.info('Excluding failed galaxies')
    valid_galaxies = []
    valid_galaxy_indices = []
    for galaxy_n in range(len(all_samples)):
        galaxy = all_samples[galaxy_n]
        logging.info('{} total samples found'.format(galaxy.shape[0]))
        assert not np.isnan(galaxy).any()
        # indices_with_nan = np.isnan(galaxy).any(axis=0)
        # galaxy_without_nans = galaxy[~indices_with_nan]
        # logging.info('{} valid samples remaining'.format(galaxy_without_nans.shape[0]))
        # if len(galaxy_without_nans) < len(galaxy) * 0.5:
            # logging.info('Excluding galaxy for low non-nan fraction: {} of {}'.format(len(galaxy_without_nans), len(galaxy)))
        if low_acceptance_suspected(galaxy):
            logging.info('Excluding galaxy for low acceptance ratio')
        else:
            valid_galaxies.append(galaxy)
            valid_galaxy_indices.append(galaxy_n)

    if len(valid_galaxy_indices) == 0:
        raise ValueError('No succesful galaxies found?')
    else:
        logging.info('{} galaxies remaining'.format(len(valid_galaxy_indices)))

    return valid_galaxies, np.array(valid_galaxy_indices)


def low_acceptance_suspected(samples, threshold=0.3):
    # expects 0th dim to be samples, 1st to be params
    # print(samples)
    # exit()
    return len(np.unique(samples, axis=0)) < len(samples) * threshold


def get_rhat(samples):
    assert len(samples.shape) == 2  # should be (sample, chain) shape, same parameter
    return arviz.rhat(np.swapaxes(samples, 0, 1))  # arxiv convention is (chain, sample) not (sample, chain) like me


def get_geweke(samples):
    assert len(samples.shape) == 2  # should be (sample, chain) shape, same parameter
    return arviz.geweke(samples.flatten())


def get_convergence_metrics(galaxies, n_params):
    for param_n in range(n_params):
        all_rhats = np.zeros((len(galaxies), n_params))
        all_gewekes = np.zeros((len(galaxies), n_params, 20))
        for galaxy_n, galaxy in enumerate(galaxies):
            samples_of_param = galaxy[:, :, param_n]
            # TODO can probably calculate these for all params at once via arviz
            rhat = get_rhat(samples_of_param)
            geweke = get_geweke(samples_of_param)
            all_rhats[galaxy_n, param_n] = rhat
            all_gewekes[galaxy_n, param_n] = geweke[:, 1]  # index 1 is the gewecke values, 0 is the trace indices
    return all_rhats, all_gewekes

def visualise_convergence_metrics(rhats, geweckes):
    pass  # not yet implemented

def check_parameter_bias(galaxies, true_params):
    params = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']

    # TODO move out to main
    # rhats, gewekes = get_convergence_metrics(galaxies, n_params=len(params))
    # visualise_convergence_metrics(rhats, geweckes)

    # fig, axes = plot_error_bars_vs_truth(params, galaxies, true_params)
    fig, axes = plot_posterior_stripes(params, galaxies, true_params)
    return fig, axes

def plot_error_bars_vs_truth(params, galaxies, true_params):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    all_axes = [ax for col in axes for ax in col]
    for param_n in range(len(params)):
        best_guesses = []
        y_errors = []
        ax = all_axes[param_n]
        for galaxy in galaxies:
            samples_of_param = galaxy[:, :, param_n]
            best_guess = np.median(samples_of_param)
            quantiles = np.quantile(samples_of_param, q=[0.1, 0.9])  # 80% credible interval
            y_error = np.abs(quantiles - best_guess)
            best_guesses.append(best_guess)
            y_errors.append(y_error)

        # colors = cm.magma(all_rhats)
        colors = None
        ax.errorbar(true_params[:, param_n], best_guesses, yerr=np.swapaxes(np.array(y_errors), 0, 1), fmt='none', c='black', zorder=1)
        ax.scatter(true_params[:, param_n], best_guesses, c=colors, zorder=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title('{}'.format(params[param_n]))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Truth')
        ax.set_ylabel(r'Best guess, 80% credible interval')
    fig.tight_layout()
    return fig, axes


def plot_posterior_stripes(params, galaxies, true_params, n_param_bins=50, n_posterior_bins=50):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    all_axes = [ax for col in axes for ax in col]
    sns.set_context('notebook')
    sns.set_style('white')

    # with a made up array, get the bins to use
    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=n_param_bins)
    _, posterior_bins = np.histogram(dummy_array, range=(0., 1.), bins=n_posterior_bins)

    for param_n in range(len(params)):
        ax = all_axes[param_n]
        posterior_record = np.zeros((n_param_bins, n_posterior_bins)) * np.nan
        for galaxy_n, galaxy in enumerate(galaxies):
            samples_of_param = galaxy[:, :, param_n].flatten()
            true_param = true_params[galaxy_n, param_n]
            true_param_index = np.digitize(true_param, param_bins)  # find the bin index for true_param
            stripe, _ = np.histogram(samples_of_param, density=True, bins=posterior_bins)
            posterior_record[true_param_index] = stripe  # currently will only show the latest, should do nan-safe mean
        ax.pcolormesh(param_bins, posterior_bins, np.transpose(posterior_record), cmap='Blues')
        ax.grid(False)
        ax.plot([0., 1.], [0., 1.], 'k--', alpha=0.3)
        ax.set_title('{}'.format(params[param_n]))
        ax.set_xlabel('Truth')
        ax.set_ylabel(r'Sampled Posterior')
    fig.tight_layout()
    return fig, axes


def write_corner_plots(galaxies, save_dir):
    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    for n, galaxy in tqdm(enumerate(galaxies)):
        figure = corner.corner(galaxy.reshape(-1, 7), labels=labels)  # middle dim is per chain
        figure.savefig(os.path.join(save_dir, 'galaxy_{}_corner.png'.format(n)))


def main(samples_dir, save_corner):

    # if not os.path.isfile(os.path.join(samples_dir, 'all_virtual.h5')):
    #     run_sampler.aggregate_performance(samples_dir, 6000, 96)  # TODO

    logging.info('Loading samples')
    samples, true_params, _ = run_sampler.read_performance(samples_dir)  # zeroth index of samples is galaxy, first is sample, second is params
    logging.info('{} galaxies found'.format(len(samples)))

    valid_galaxies, valid_galaxy_indices = get_valid_galaxy_indices(samples)
    valid_true_params = true_params[valid_galaxy_indices]

    # print(valid_galaxies.shape, valid_true_params.shape)

    save_dir = os.path.join(samples_dir, 'evaluation')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    logging.info('Checking parameter bias')
    fig, _ = check_parameter_bias(valid_galaxies, valid_true_params)
    fig.savefig(os.path.join(save_dir, 'parameter_bias.pdf'))
    fig.savefig(os.path.join(save_dir, 'parameter_bias.png'))

    if save_corner:  # optional as somewhat slow
        write_corner_plots(valid_galaxies, save_dir)


if __name__ == '__main__':

    """
    Check if the emulated HMC sampling is correctly recovering the original galaxy parameters for the forward model.

    Example use:
    python agnfinder/tf_sampling/evaluate_performance.py --save-dir results/emulated_sampling/latest_6000_96_optimised
    """

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--corner', dest='save_corner', action='store_true', default=False)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    main(args.save_dir, args.save_corner)
