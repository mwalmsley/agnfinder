import glob
import logging
import argparse

import tqdm
import numpy as np
from agnfinder.tf_sampling import run_sampler
import h5py

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--bins', dest='bins', default=50)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=args.bins)

    galaxy_locs = glob.glob(args.save_dir + '/galaxy*.h5')
    assert galaxy_locs
    
    for galaxy_loc in tqdm.tqdm(galaxy_locs):
        marginals = np.zeros((7, 50))
        for param_n in range(7):
            samples, _, _ = run_sampler.read_h5(galaxy_loc)
            param_marginals, _ = np.histogram(samples[:, :, param_n], density=True, bins=param_bins)
            marginals[param_n] = param_marginals
        f = h5py.File(galaxy_loc, mode='r+')
        try:
            del f['marginals']
        except KeyError:
            pass
        marginal_dataset = f.create_dataset('marginals', data=marginals)
