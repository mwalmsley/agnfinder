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


def main(save_dir, use_filter, max_redshift):
    params, marginals, true_params, _ = load_samples(save_dir, use_filter, max_redshift)  # don#t need samples themselves
    posterior_records, param_bins = get_all_posterior_records(marginals, true_params, n_param_bins=50, n_posterior_bins=50)
    fig, axes = plot_posterior_stripes(posterior_records, param_bins, params)
    return fig, axes


def get_all_posterior_records(marginals, true_params, n_param_bins, n_posterior_bins):
    # with a made up array, get the bins to use
    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=n_param_bins)

    posterior_records = []
    if true_params[:, -1].mean() > 1.:  # is scale param
        n_params = true_params.shape[1] - 1
    else:
        n_params = true_params.shape[1]
    for param_n in range(n_params):  # now excluding scale
        print(param_n)
        posterior_record = get_posterior_record(marginals, true_params, param_n, param_bins, n_param_bins, n_posterior_bins)
        posterior_records.append(posterior_record)
    return posterior_records, param_bins


def get_posterior_record(marginals, true_params, param_n, param_bins, n_param_bins, n_posterior_bins):
    galaxy_counts = np.zeros((n_param_bins))  # to track how many galaxies have been added
    posterior_record = np.zeros((n_param_bins, n_posterior_bins)) * 0
    for galaxy_n, _ in enumerate(marginals):
        true_param = true_params[galaxy_n, param_n]
        true_param_index = np.digitize(true_param, param_bins)  # fnd the bin index for true_param

        stripe = marginals[galaxy_n, param_n]
        if true_param_index < n_param_bins:  # exclude =50 edge case TODO
            posterior_record[true_param_index] += np.nan_to_num(stripe)  # nans to 0's

            galaxy_counts[true_param_index] += 1

    # divide out by how many galaxies were added at each index
    # posterior_record = posterior_record / galaxy_counts
    for n in range(len(galaxy_counts)):
        posterior_record[n] = posterior_record[n] / galaxy_counts[n]
    print(posterior_record)
    # replace any 0's with nans, for clarity
    posterior_record[np.isclose(posterior_record, 0)] = np.nan
    # trim extreme values
    ceiling = np.quantile(posterior_record[~np.isnan(posterior_record)], .95)
    posterior_record = np.clip(posterior_record, 0, ceiling)
    return posterior_record


def plot_posterior_stripes(posterior_records, param_bins, params):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    all_axes = [ax for col in axes for ax in col]
    sns.set_context('notebook')
    sns.set_style('white')
    for param_n, record in enumerate(posterior_records):
        ax = all_axes[param_n]
        ax.pcolormesh(param_bins, param_bins, np.transpose(record), cmap='Blues')  
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
    model_params = ['redshift', 'mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling', 'inclination', 'agn_mass', 'agn_torus_mass', 'zred', 'scale']
    human_names = ['Redshift', 'Stellar Mass', 'Dust', 'Age', 'Tau', 'AGN Disk Scale', 'AGN E(B-V)', 'AGN Torus Scale', 'AGN Torus Incl.', 'AGN Disk Scale', 'AGN Torus Scale', 'Redshift', 'Scale']
    renamer = dict(zip(model_params, human_names))
    return [renamer[x] for x in input_names]

def load_samples(save_dir, use_filter, max_redshift, min_acceptance=0.0, frac_to_load=25):
    galaxy_locs = glob.glob(save_dir + '/*.h5')
    assert galaxy_locs

    # open one file to work out the format of the data
    for n, galaxy_loc in tqdm(enumerate(galaxy_locs), unit=' galaxies loaded'):
        f = h5py.File(galaxy_locs[0], mode='r')
        params = f['samples'].attrs['free_param_names']
        # don't care about fixed params
    params = rename_params(params)

    all_samples = []
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
        samples = np.squeeze(f['samples'][...])  # okay to load, will not keep
        all_samples.append(samples[::frac_to_load])  # only loading 1 in 25 samples!
        if use_filter:
            successful_run[n] = within_percentile_limits(samples)
        else:
            successful_run[n] = True
        
        # successful_run[n] = True  # disable for now
        if 'Redshift' not in params:
            allowed_redshift[n] = f['fixed_params'][0] * 4 < max_redshift  # absolutely must match hypercube physical redshift limit
        else:
            allowed_redshift[n] = True

        marginals[n] = galaxy_marginals
        true_params[n] = galaxy_true_params

    # filter to galaxies with decent acceptance
    logging.info('{} galaxies of {} have mean acceptance > {}'.format(allowed_acceptance.sum(), len(allowed_acceptance), min_acceptance))
    logging.info('{} galaxies of {} have redshift > {}'.format(allowed_redshift.sum(), len(allowed_redshift), max_redshift))
    logging.warning('{} galaxies of {} are successful'.format(successful_run.sum(), len(successful_run)))
    # accept = allowed_acceptance & allowed_redshift & successful_run
    # all_samples = np.array(all_samples)
    # all_samples = np.array(all_samples)[accept]
    # marginals = marginals[accept]
    # true_params = true_params[accept]
    return params, marginals, true_params, all_samples


def percentile_spreads(samples, quantile_width=50):
    upper_q = int(50+quantile_width/2)
    lower_q = int(50-quantile_width/2)
    assert upper_q > lower_q
    assert upper_q < 100
    assert lower_q > 0
    return np.percentile(samples, upper_q, axis=0) - np.percentile(samples, lower_q, axis=0)


def within_percentile_limits(samples, limits=None):
    # if limits is None:
    #     limits = np.array([0.00415039, 0.00977203, 0.00708008, 0.00683642, 0.00488902, 0.00097656, 0.00684875, 0.01074265])  # warning, cube dependent

    if samples.ndim > 2: # flatten the chains
        samples = samples.reshape(-1, samples.shape[2])

    pcs_10 = percentile_spreads(samples, quantile_width=10)  # 1D array of percentile spread by param
    pcs_25 = percentile_spreads(samples, quantile_width=25)

    if np.any(pcs_10 > 1.) or np.any(pcs_25 > 1.):  # spread is somehow outside allowed range -> bad sample -> reject
        return False



    # good_dust2 = valid_pcs_25[:, 1] > 0.01
    good_tau = pcs_25[3] > 0.006
    good_agn_extinction = pcs_10[5] > 0.011
    # good_agn_torus= pcs_10[6] > 0.02
    # good_inclination = pcs_10[7] > 0.02

    return good_agn_extinction & good_tau
    # return ~(good_agn_extinction & good_tau)  # flip if you like, to debug

    # return compare_percentiles_with_limits(valid_pcs, limits)

def compare_percentiles_with_limits(valid_pcs, limits):
    return np.squeeze(np.sum(valid_pcs < limits, axis=1) < 2.)  # no more than 1 parameter can have less 75%-25% spread than the limits (set to discard 15% of data)


if __name__ == '__main__':

    sns.set_context('notebook')
    sns.set(font_scale=1.3)

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--raw', dest='raw', default=False, action='store_true')
    # parser.add_argument('--min-acceptance', default=0.6, type=float, dest='min_acceptance')
    parser.add_argument('--max-redshift', type=float, dest='max_redshift', default=4.0)
    args = parser.parse_args()

    # more convenient to only specify when you *don't* want a filter
    use_filter = True
    if args.raw:
        use_filter = False

    fig, axes = main(args.save_dir, use_filter, args.max_redshift)

    fig.savefig('results/latest_posterior_stripes.png')
    fig.savefig('results/latest_posterior_stripes.pdf')

    # python agnfinder/tf_sampling/parameter_recovery.py --save-dir results/emulated_sampling/latest_emcee_5000_10000_1_optimised  --raw
    # python agnfinder/tf_sampling/parameter_recovery.py --save-dir results/emulated_sampling/latest_hmc_10000_40000_16_optimised  --raw
    # python agnfinder/tf_sampling/parameter_recovery.py --save-dir results/vanilla_emcee  --raw