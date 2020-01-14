import os
import logging
import argparse
import json  # temporary, for debugging

from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf  # just for eager toggle
# os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'

from agnfinder.tf_sampling import deep_emulator, api, hmc

# TODO change indices to some kind of unique id, perhaps? will need for real galaxies...
def sample_galaxy_batch(names, true_observation, true_params, emulator, n_burnin, n_samples, n_chains, init_method, save_dir):
    assert len(true_observation.shape) == 2
    assert len(true_params.shape) == 2
    assert len(names) == true_params.shape[0]
    assert true_observation.max() < 1e-3  # should be denormalised i.e. actual photometry in maggies

    problem = api.SamplingProblem(true_observation, true_params, forward_model=emulator)
    sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
    samples, successfully_adapted = sampler()

    assert samples.shape[0] == n_samples
    assert samples.shape[1] == np.sum(successfully_adapted)
    assert samples.shape[2] == true_params.shape[1]

    names_were_adapted = dict(zip(names, successfully_adapted))  # dicts are ordered in 3.7+ I think
    remaining_names = [k for k, v in names_were_adapted.items() if v]
    discarded_names = [k for k, v in names_were_adapted.items() if not v]

    if len(discarded_names) != 0:
        logging.warning('Galaxies {} did not adapt and were discarded'.format(discarded_names))
    else:
        logging.info('All galaxies adapted succesfully')

    for galaxy_n, name in enumerate(remaining_names):
        save_file = get_galaxy_save_file(name, save_dir)
        f = h5py.File(save_file, mode='w')  # will overwrite
        galaxy_samples = np.expand_dims(samples[:, galaxy_n], axis=1)
        f.create_dataset('samples', data=galaxy_samples)  # leave the chain dimension as 1 for now
        f.create_dataset('true_observations', data=true_observation[galaxy_n])
        
        if true_params is not None:
            f.create_dataset('true_params', data=true_params[galaxy_n])

        marginal_bins = 50
        dummy_array = np.zeros(42)  # anything
        _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=marginal_bins)
        marginals = np.zeros((7, marginal_bins))
        for param_n in range(7):
            marginals[param_n], _ = np.histogram(galaxy_samples[:, :, param_n], density=True, bins=param_bins)
        f.create_dataset('marginals', data=marginals)


def get_galaxy_save_file(i, save_dir):
    return os.path.join(save_dir, 'galaxy_{}_performance.h5'.format(i))


def aggregate_performance(save_dir, n_samples, chains_per_galaxy):
    logging.info(f'Aggregating galaxies in {save_dir}')
    assert chains_per_galaxy == 1  # TODO remove as arg?
    logging.debug('Creating virtual dataset')
    performance_files = [os.path.join(save_dir, x) for x in os.listdir(save_dir) if x.endswith('_performance.h5')]
    n_sources = len(performance_files)
    logging.info('Using source files: {} (max 10 shown)'.format(performance_files[:10]))
    logging.debug('Specifing expected data')
    samples_vl = h5py.VirtualLayout(shape=(n_sources, n_samples, chains_per_galaxy, 7), dtype='f')
    true_params_vl = h5py.VirtualLayout(shape=(n_sources, 7), dtype='f')
    true_observations_vl = h5py.VirtualLayout(shape=(n_sources, 12), dtype='f')

    samples_source_shape = (n_samples, chains_per_galaxy, 7)
    logging.info('shape of samples expected: {}'.format(samples_source_shape))

    logging.debug('Specifying sources')
    for i, file_loc in enumerate(performance_files):
        assert os.path.isfile(file_loc)

        samples_vl[i] = h5py.VirtualSource(file_loc, 'samples', shape=samples_source_shape)
        true_params_vl[i] = h5py.VirtualSource(file_loc, 'true_params', shape=(7,))
        true_observations_vl[i] = h5py.VirtualSource(file_loc, 'true_observations', shape=(12,))

    # Add virtual dataset to output file
    logging.debug('Writing virtual dataset')
    with h5py.File(aggregate_filename(save_dir), 'w') as f:
        f.create_virtual_dataset('samples', samples_vl, fillvalue=0)
        f.create_virtual_dataset('true_params', true_params_vl, fillvalue=0)
        f.create_virtual_dataset('true_observations', true_observations_vl, fillvalue=0)
    

def read_performance(save_dir):
    # if the code hangs while reading, there's a shape mismatch between sources and virtual layout - probably samples.
    file_loc = aggregate_filename(save_dir)
    return read_h5(file_loc)

def read_h5(file_loc):
    with h5py.File(file_loc, 'r') as f:
        logging.debug('Reading {}'.format(file_loc))
        logging.debug(list(f.keys()))
        samples = f['samples'][...]
        logging.debug('Samples read')
        true_params = f['true_params'][...]
        true_observations = f['true_observations'][...]
    logging.debug('{} loaded'.format(file_loc))
    return samples, true_params, true_observations


def aggregate_filename(save_dir):
    return os.path.join(save_dir, 'all_virtual.h5')
