import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

from agnfinder.tf_sampling import parameter_recovery


def load_galaxies(galaxy_locs, quantile_spreads=None):  # stripped down version, just need samples -> percentile spreads

    # open one galaxy for params
    with h5py.File(galaxy_locs[0], mode='r') as f:
        params = f['samples'].attrs['free_param_names']

    if quantile_spreads is None:
            quantile_spreads = [10, 25]

    # must match run_sampler.py
    marginal_bins = 50
    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=marginal_bins)

    galaxies = []
    for galaxy_loc in galaxy_locs:
        try:
            with h5py.File(galaxy_loc, mode='r') as f:
                galaxy = {
                    'true_observations': np.squeeze(f['true_observations'][...]),
                    'true_params': np.squeeze(f['true_params'][...]),
                    'uncertainty': f['uncertainty'][...],
                    'samples': np.squeeze(f['samples'][::100])
                }

                samples = np.squeeze(f['samples'][...])
                for spread in quantile_spreads:
                    galaxy[f'percentile_spreads_{spread}'] = parameter_recovery.percentile_spreads(samples, spread)

                galaxy['marginals'] = f['marginals'][...]
                galaxy['marginal_bins'] = param_bins

                galaxies.append(galaxy)
        except OSError:
            pass
    return galaxies, params


def get_surprise(galaxy):
    p_by_param = np.zeros(len(galaxy['true_params']))
    for param_n, true_param in enumerate(galaxy['true_params']):
        bin_index = np.digitize(true_param, galaxy['marginal_bins'])
        if bin_index == 50:  # above the bin range?
            bin_index -= 1
        p_of_index = galaxy['marginals'][param_n, bin_index]
        p_by_param[param_n] = p_of_index
    return p_by_param


def galaxy_to_row(galaxy, quantile_spreads=None):

    if quantile_spreads is None:
            quantile_spreads = [10, 25]

    bands = ['u_sloan', 'g_sloan', 'r_sloan', 'i_sloan', 'z_sloan', 'VISTA_H',
        'VISTA_J', 'VISTA_Y']
    errors = ['u_sloan_err', 'g_sloan_err', 'r_sloan_err', 'i_sloan_err',
            'z_sloan_err', 'VISTA_H_err', 'VISTA_J_err',
            'VISTA_Y_err']

    row = {}
    for param_n in range(len(params)):
        param = params[param_n]
        row[param] = np.median(galaxy['samples'], axis=0)[param_n]
        for spread in quantile_spreads:
            row[f'{param}_pc_{spread}'] = galaxy[f'percentile_spreads_{spread}'][param_n]
    for band_n in range(len(bands)):
        row[bands[band_n]] = galaxy['true_observations'][band_n]
        row[errors[band_n]] = galaxy['uncertainty'][band_n]
    return row


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='results/emulated_sampling/latest_80000_512_optimised')
    # parser.add_argument('--min-acceptance', default=0.6, type=float, dest='min_acceptance')
    # parser.add_argument('--max-redshift', type=float, dest='max_redshift', default=4.0)
    args = parser.parse_args()

    all_galaxy_locs = glob.glob(args.save_dir + '/galaxy*.h5')  # these must NOT have been run with success checks, or should be filtered to first chain only
    assert all_galaxy_locs

    # filter to first chain only
    galaxy_locs = [x for x in all_galaxy_locs if '_0.h5' in x]
    assert galaxy_locs

    galaxies, params = load_galaxies(galaxy_locs)

    surprise = np.array([get_surprise(galaxy) for galaxy in galaxies])

    fig, axes = plt.subplots(nrows=8, figsize=(8, 16))
    for param_n in range(8):
        ax = axes[param_n]
        ax.hist(surprise[:, param_n], bins=200)
        ax.set_title(params[param_n])
        ax.set_xlim([0., 10.])
    fig.tight_layout()
    fig.savefig('latest_surprise_by_param.png')

    min_surprise = 0.0000001

    print('Bad param estimates: ', np.sum(surprise < min_surprise)/surprise.size)
    print('Any bad params: ', np.sum(np.any(surprise < min_surprise, axis=1))/len(surprise))
    print('All bad params: ', np.sum(np.all(surprise < min_surprise, axis=1))/len(surprise))
    bad_params = surprise < min_surprise
    for n_bad in range(len(params)+1):
        print(f'{n_bad} bad params: ', np.sum( np.sum(bad_params, axis=1) == n_bad)/len(surprise))

    # galaxy_is_bad = np.any(surprise < min_surprise, axis=1)
    galaxy_is_bad = np.sum(bad_params, axis=1) >=7  # i.e. 7 or 8 bad params
    print(f'Bad galaxies: {np.sum(galaxy_is_bad)} of {len(galaxy_is_bad)}')
    bad_galaxy_indices = np.arange(len(galaxies))[galaxy_is_bad]
    # good_galaxy_indices = np.arange(len(galaxies))[~galaxy_is_bad]

    data = [galaxy_to_row(galaxy) for galaxy in galaxies]
    df = pd.DataFrame(data)
    df['success'] = ~galaxy_is_bad
    df.head()

    pc_cols = [x for x in df.columns.values if 'pc_10' in x] + [x for x in df.columns.values if 'pc_25' in x]

    clipped_df = df.copy()
    for col in pc_cols:
        clipped_df = df.query(f'{col} < 1.')

    clf = DecisionTreeClassifier(max_depth=2)

    X = clipped_df[pc_cols]
    y = clipped_df['success'].values.reshape(-1)

    results = cross_validate(clf, X, y, cv=3, scoring=['recall', 'precision'])
    # # print(scores.mean()
    print(results['test_recall'].mean(), results['test_precision'].mean())

    clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(16,16))
    _ = plot_tree(clf, filled=True, ax=ax, feature_names=pc_cols)
    fig.savefig('results/latest_tree.png')



    # pcs = np.abs(np.array([x['percentile_spreads'] for x in galaxies]))
    # valid_pcs = pcs[np.all(pcs < 1., axis=1)]

    # limits = np.percentile(valid_pcs, q=args.q, axis=0)  # fiddle with q to get 85% or so within limits
    # within_limits = parameter_recovery.compare_percentiles_with_limits(valid_pcs, limits)
    # print(within_limits.mean())

    # print(limits)

