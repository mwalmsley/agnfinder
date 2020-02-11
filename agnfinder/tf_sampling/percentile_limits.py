import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import h5py

from agnfinder.tf_sampling import parameter_recovery


def load_galaxies(galaxy_locs):  # stripped down version, just need samples -> percentile spreads
    galaxies = []
    for galaxy_loc in galaxy_locs:
        try:
            with h5py.File(galaxy_loc, mode='r') as f:
                galaxy = {}
                samples = np.squeeze(f['samples'][...])
                galaxy['percentile_spreads'] = parameter_recovery.percentile_spreads(samples)
                galaxies.append(galaxy)
        except OSError:
            pass
    return galaxies


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='results/emulated_sampling/latest')
    parser.add_argument('--q', dest='q', type=int, default=10)
    parser.add_argument('--min-acceptance', default=0.6, type=float, dest='min_acceptance')
    parser.add_argument('--max-redshift', type=float, dest='max_redshift', default=4.0)
    args = parser.parse_args()

    assert 0 < args.q < 100

    all_galaxy_locs = glob.glob(args.save_dir + '/galaxy*.h5')  # these must NOT have been run with success checks, or should be filtered to first chain only
    assert all_galaxy_locs

    # filter to first chain only
    galaxy_locs = [x for x in all_galaxy_locs if '_0.h5' in x]
    assert galaxy_locs

    galaxies = load_galaxies(galaxy_locs)

    pcs = np.abs(np.array([x['percentile_spreads'] for x in galaxies]))
    valid_pcs = pcs[np.all(pcs < 1., axis=1)]

    limits = np.percentile(valid_pcs, q=args.q, axis=0)  # fiddle with q to get 85% or so within limits
    within_limits = parameter_recovery.compare_percentiles_with_limits(valid_pcs, limits)
    print(within_limits.mean())

    print(limits)

