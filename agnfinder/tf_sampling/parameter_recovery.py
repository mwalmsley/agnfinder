import argparse
import logging
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm
import h5py
from tqdm import tqdm

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

    for param_n in range(len(params)):
        ax = all_axes[param_n]
        posterior_record = np.zeros((n_param_bins, n_posterior_bins)) * 0
        posterior_colors = np.zeros((n_param_bins, n_posterior_bins, 4)) * 0
        galaxy_counts = np.zeros((n_param_bins))  # to track how many galaxies have been added
        for galaxy_n, _ in enumerate(marginals):
            true_param = true_params[galaxy_n, param_n]
            true_param_index = np.digitize(true_param, param_bins)  # fnd the bin index for true_param
            stripe = marginals[galaxy_n, param_n]
            if true_param_index < n_param_bins:  # exclude =50 edge case TODO
                posterior_record[true_param_index] += np.nan_to_num(stripe)  # nans to 0's
                # posterior_record[true_param_index] = stripe
                galaxy_counts[true_param_index] += 1

        print(posterior_record)
        # divide out by how many galaxies were added at each index
        posterior_record = posterior_record / galaxy_counts
        # replace any 0's with nans, for clarity
        posterior_record[np.isclose(posterior_record, 0)] = np.nan
        # trim extreme values
        ceiling = np.quantile(posterior_record[~np.isnan(posterior_record)], .98)
        posterior_record = np.clip(posterior_record, 0, ceiling)

        # plot in single color
        ax.pcolormesh(param_bins, param_bins, np.transpose(posterior_record), cmap='Blues')  
        # OR plot colored by value of one param
        # coloring_param_index = 5
        # for galaxy_n, galaxy in enumerate(marginals):
        #     custom_cmap = get_cmap(true_params[galaxy_n, coloring_param_index])
        #     true_param = true_params[galaxy_n, param_n]
        #     true_param_index = np.digitize(true_param, param_bins)  # find the bin index for true_param
        #     if true_param_index < n_param_bins:  # exclude =50 edge case TODO
        #         posterior_colors[true_param_index] = custom_cmap(posterior_record[true_param_index])
        # ax.imshow(np.transpose(posterior_colors, axes=[1, 0, 2]), origin='lower')
        
        ax.grid(False)
        # ax.plot([0., 50.], [0., 50.], 'k--', alpha=0.3)
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

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    params = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']

    galaxy_locs = glob.glob(args.save_dir + '/galaxy*.h5')
    assert galaxy_locs
    
    marginals = np.zeros((len(galaxy_locs), len(params), 50))
    true_params = np.zeros((len(galaxy_locs), len(params)))
    for n, galaxy_loc in tqdm(enumerate(galaxy_locs), unit=' galaxies loaded'):
        f = h5py.File(galaxy_loc, mode='r')
        galaxy_marginals = f['marginals'][...]
        galaxy_true_params = f['true_params'][...]
        marginals[n] = galaxy_marginals
        true_params[n] = galaxy_true_params

    fig, axes = plot_posterior_stripes(params, marginals, true_params)
    fig.savefig('results/latest_posterior_stripes.png')
    # plt.gcf()
    # plt.show()