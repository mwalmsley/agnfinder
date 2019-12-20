import argparse
import logging
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm
import h5py

def check_parameter_bias(galaxies, true_params):
    params = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']

    # TODO move out to main
    # rhats, gewekes = get_convergence_metrics(galaxies, n_params=len(params))
    # visualise_convergence_metrics(rhats, geweckes)

    # fig, axes = plot_error_bars_vs_truth(params, galaxies, true_params)
    fig, axes = plot_posterior_stripes(params, galaxies, true_params)
    return fig, axes


def plot_posterior_stripes(params, marginals, true_params, n_param_bins=50, n_posterior_bins=50):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    all_axes = [ax for col in axes for ax in col]
    sns.set_context('notebook')
    sns.set_style('white')

    # with a made up array, get the bins to use
    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=n_param_bins)

    coloring_param_index = 6
    for param_n in range(len(params)):
        ax = all_axes[param_n]
        posterior_record = np.zeros((n_param_bins, n_posterior_bins)) * np.nan
        posterior_colors = np.zeros((n_param_bins, n_posterior_bins, 4)) * np.nan
        for galaxy_n, galaxy in enumerate(marginals):
            true_param = true_params[galaxy_n, param_n]
            true_param_index = np.digitize(true_param, param_bins)  # find the bin index for true_param
            stripe = marginals[galaxy_n, param_n]
            if true_param_index < n_param_bins:  # exclude =50 edge case TODO
                posterior_record[true_param_index] = stripe  # currently will only show the latest, should do nan-safe mean

        posterior_record = posterior_record / np.max(posterior_record[~np.isnan(posterior_record)])

        # ax.pcolormesh(param_bins, posterior_bins, np.transpose(posterior_record), cmap='Blues')  
        for galaxy_n, galaxy in enumerate(marginals):
            custom_cmap = get_cmap(true_params[galaxy_n, coloring_param_index])
            true_param = true_params[galaxy_n, param_n]
            true_param_index = np.digitize(true_param, param_bins)  # find the bin index for true_param
            if true_param_index < n_param_bins:  # exclude =50 edge case TODO
                posterior_colors[true_param_index] = custom_cmap(posterior_record[true_param_index])
        ax.imshow(np.transpose(posterior_colors, axes=[1, 0, 2]), origin='lower')
        
        ax.grid(False)
        ax.plot([0., 50.], [0., 50.], 'k--', alpha=0.3)
        ax.set_title('{}'.format(params[param_n]))
        ax.set_xlabel('Truth')
        ax.set_ylabel(r'Sampled Posterior')
    fig.tight_layout()
    return fig, axes

def get_cmap(hue_val):
    base_color = plt.get_cmap('viridis')(hue_val)[:3]
    base_colors = np.array([base_color for _ in np.linspace(0, 1, 256)])
    x = np.linspace(0, 1, 256).reshape(-1, 1)
    newcolors = np.concatenate((base_colors, x), axis=1)
    return ListedColormap(newcolors)

if __name__ == '__main__':
    """
    Example use:
    python agnfinder/tf_sampling/parameter_recovery.py --save-dir results/extra_filters/latest_6000_96_optimised
    """

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    params = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']

    galaxy_locs = glob.glob(args.save_dir + '/galaxy*.h5')
    assert galaxy_locs
    
    marginals = np.zeros((len(galaxy_locs), len(params), 50))
    true_params = np.zeros((len(galaxy_locs), len(params)))
    for n, galaxy_loc in enumerate(galaxy_locs):
        f = h5py.File(galaxy_loc, mode='r')
        galaxy_marginals = f['marginals'][...]
        galaxy_true_params = f['true_params'][...]
        marginals[n] = galaxy_marginals
        true_params[n] = galaxy_true_params

    fig, axes = plot_posterior_stripes(params, marginals, true_params)
    fig.tight_layout()
    # plt.gcf()
    plt.show()