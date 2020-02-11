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



def plot_posterior_stripes(params, marginals, true_params, n_param_bins=50, n_posterior_bins=50):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
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
            # print(true_param_index)
            stripe = marginals[galaxy_n, param_n]
            if true_param_index < n_param_bins:  # exclude =50 edge case TODO
                posterior_record[true_param_index] += np.nan_to_num(stripe)  # nans to 0's
                # posterior_record[true_param_index] = stripe
                galaxy_counts[true_param_index] += 1

        # print(posterior_record[:, 0])
        # print(posterior_record)
        # divide out by how many galaxies were added at each index
        # posterior_record = posterior_record / galaxy_counts
        for n in range(len(galaxy_counts)):
            posterior_record[n] = posterior_record[n] / galaxy_counts[n]
        # print(posterior_record)
        # replace any 0's with nans, for clarity
        posterior_record[np.isclose(posterior_record, 0)] = np.nan
        # trim extreme values
        ceiling = np.quantile(posterior_record[~np.isnan(posterior_record)], .95)
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
        ax.set_title('{}'.format(params[param_n]), fontsize=22)
        ax.set_xlabel('Truth')
        ax.set_ylabel(r'Sampled Posterior')
    for ax_n, ax in enumerate(all_axes):
        if ax_n >= len(params):
            ax.remove()
    fig.tight_layout()
    return fig, axes


def get_cmap(hue_val):
    base_color = plt.get_cmap('viridis')(hue_val)[:3]
    base_colors = np.array([base_color for _ in np.linspace(0, 1, 256)])
    x = np.linspace(0, 1, 256).reshape(-1, 1)
    newcolors = np.concatenate((base_colors, x), axis=1)
    return ListedColormap(newcolors)


def rename_params(input_names):
    model_params = ['redshift', 'mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling', 'inclination']
    human_names = ['Redshift', 'Stellar Mass', 'Dust', 'Age', 'Tau', 'AGN Disk Scale', 'AGN E(B-V)', 'AGN Torus Scale', 'AGN Torus Incl.']
    renamer = dict(zip(model_params, human_names))
    return [renamer[x] for x in input_names]

def load_samples(save_dir, min_acceptance, max_redshift):
    galaxy_locs = glob.glob(save_dir + '/*.h5')
    assert galaxy_locs

    # open one file to work out the format of the data
    for n, galaxy_loc in tqdm(enumerate(galaxy_locs), unit=' galaxies loaded'):
        f = h5py.File(galaxy_locs[0], mode='r')
        params = f['samples'].attrs['free_param_names']
        # don't care about fixed params
    params = rename_params(params)

    marginals = np.zeros((len(galaxy_locs), len(params), 50))  # TODO magic number which must match run_sampler.py
    true_params = np.zeros((len(galaxy_locs), len(params)))
    allowed_redshift = np.zeros(len(galaxy_locs), dtype=bool)
    allowed_acceptance = np.zeros(len(galaxy_locs), dtype=bool)
    successful_run = np.zeros(len(galaxy_locs), dtype=bool)
    for n, galaxy_loc in tqdm(enumerate(galaxy_locs), unit=' galaxies loaded'):
        f = h5py.File(galaxy_loc, mode='r')
        galaxy_marginals = f['marginals'][...]
        galaxy_true_params = f['true_params'][...]
        # is_accepted = f['is_accepted'][...].mean()
        # accept[n] = is_accepted >= args.min_acceptance
        value_for_80p = np.quantile(galaxy_marginals, .8, axis=1)
        num_geq_80p = (galaxy_marginals.transpose() > value_for_80p).sum(axis=0)
        # print(num_geq_80p, num_geq_80p.shape)
        allowed_acceptance[n] = np.mean(num_geq_80p) > min_acceptance
        samples = f['samples'][...] # okay to load, will not keep
        successful_run[n] = within_percentile_limits(samples)
        if 'Redshift' not in params:
            allowed_redshift[n] = f['fixed_params'][0] * 4 < max_redshift  # absolutely must match hypercube physical redshift limit
        else:
            allowed_redshift[n] = True

        marginals[n] = galaxy_marginals
        true_params[n] = galaxy_true_params

    # filter to galaxies with decent acceptance
    logging.info('{} galaxies of {} have mean acceptance > {}'.format(allowed_acceptance.sum(), len(allowed_acceptance), min_acceptance))
    logging.info('{} galaxies of {} have redshift > {}'.format(allowed_redshift.sum(), len(allowed_redshift), max_redshift))
    logging.info('{} galaxies of {} are successful'.format(successful_run.sum(), len(successful_run)))
    accept = allowed_acceptance & allowed_redshift & successful_run
    marginals = marginals[accept]
    true_params = true_params[accept]
    return params, marginals, true_params


def percentile_spreads(samples):
    return np.percentile(samples, 75, axis=0) - np.percentile(samples, 25, axis=0)

def within_percentile_limits(samples):
    limits = np.array([0.02932622, 0.07219234, 0.03350993, 0.05405632, 0.03579117,
       0.03457421, 0.03837388, 0.05567279])
    pcs = percentile_spreads(samples)
    valid_pcs = pcs[np.all(pcs < 1., axis=1)]
    return bool(np.sum(valid_pcs < limits, axis=1) < 2.)  # no more than 1 parameter can have less 75%-25% spread than the limits (set to discard 15% of data)

def main(save_dir, min_acceptance, max_redshift):
    params, marginals, true_params = load_samples(save_dir, min_acceptance, max_redshift)
    fig, axes = plot_posterior_stripes(params, marginals, true_params)
    return fig, axes


if __name__ == '__main__':

    sns.set_context('notebook')
    sns.set(font_scale=1.3)

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--min-acceptance', default=0.6, type=float, dest='min_acceptance')
    parser.add_argument('--max-redshift', type=float, dest='max_redshift', default=4.0)
    args = parser.parse_args()

    fig, axes = main(args.save_dir, args.min_acceptance, args.max_redshift)

    fig.savefig('results/latest_posterior_stripes.png')
    fig.savefig('results/latest_posterior_stripes.pdf')