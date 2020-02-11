import os
import logging
import argparse
import json  # temporary, for debugging

from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf  # just for eager toggle

from agnfinder.tf_sampling import deep_emulator, api, hmc, nested

# TODO change indices to some kind of unique id, perhaps? will need for real galaxies...
def sample_galaxy_batch(galaxy_ids, true_observation, fixed_params, uncertainty, true_params, emulator, n_burnin, n_samples, n_chains, init_method, save_dir, free_param_names, fixed_param_names):
    assert len(true_observation.shape) == 2
    assert len(true_params.shape) == 2
    assert true_params.shape[1] == len(free_param_names)
    assert fixed_params.shape[1] == len(fixed_param_names)
    assert len(galaxy_ids) == true_params.shape[0]
    if true_observation.max() < 1e-3:  # should be denormalised i.e. actual photometry in maggies
        logging.info('True observation is {}'.format(true_observation))
        logging.critical('True observation max is {} - make sure it is in maggies, not mags!'.format(true_observation))

    problem = api.SamplingProblem(true_observation, true_params, forward_model=emulator, fixed_params=fixed_params, uncertainty=uncertainty)  # will pass in soon
    
    # HMC/NUTS
    sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
    # nested sampling
    # sampler = nested.SamplerNested(problem, n_live=400)

    samples, is_successful, sample_weights, log_evidence, metadata = sampler()
    # metadata MUST be already filtered by is_successful

    # assert samples.shape[0] == n_samples  # NOT TRUE for nested sampling!
    assert samples.shape[1] == np.sum(is_successful)
    assert samples.shape[2] == true_params.shape[1]

    # filter the args to only galaxies which survived
    # samples is already filtered
    # is_accepted is already filtered
    # sample_weights is already filtered
    true_observation = true_observation[is_successful]
    true_params = true_params[is_successful]
    fixed_params = fixed_params[is_successful]
    uncertainty = uncertainty[is_successful]
    # galaxy_ids are a bit more awkward
    ids_were_adapted = dict(zip(galaxy_ids, is_successful))  # dicts are ordered in 3.7+ I think
    remaining_galaxy_ids = [k for k, v in ids_were_adapted.items() if v]
    discared_galaxy_ids = [k for k, v in ids_were_adapted.items() if not v]
    if len(discared_galaxy_ids) != 0:
        logging.warning('Galaxies {} did not adapt and were discarded'.format(discared_galaxy_ids))
    else:
        logging.info('All galaxies adapted succesfully')

    # now, galaxy_n will always line up with a remanining galaxy
    for galaxy_n, name in tqdm(enumerate(remaining_galaxy_ids), unit=' galaxies saved'):
        save_file, chain_n = get_galaxy_save_file_next_chain(name, save_dir)
        f = h5py.File(save_file, mode='w')  # will overwrite
        galaxy_samples = np.expand_dims(samples[:, galaxy_n], axis=1)
        # for 0-1 decimals, float16 is more than precise enough and much smaller files
        dset = f.create_dataset('samples', data=galaxy_samples.astype(np.float16), dtype=np.float16)  # leave the chain dimension as 1 for now
        dset.attrs['free_param_names'] = free_param_names
        dset.attrs['init_method'] = init_method
        dset.attrs['n_burnin'] = n_burnin
        dset.attrs['galaxy_id'] = name
        f.create_dataset('chain', data=chain_n)
        f.create_dataset('sample_weights', data=sample_weights[:, galaxy_n])
        f.create_dataset('log_evidence', data=log_evidence[galaxy_n])
        f.create_dataset('true_observations', data=true_observation[galaxy_n])
        dset = f.create_dataset('fixed_params', data=fixed_params[galaxy_n])
        dset.attrs['fixed_param_names'] = fixed_param_names
        f.create_dataset('uncertainty', data=uncertainty[galaxy_n])
        for key, data in metadata.items():
            f.create_dataset(key, data=data)
        if true_params is not None:
            f.create_dataset('true_params', data=true_params[galaxy_n])

        marginal_bins = 50
        dummy_array = np.zeros(42)  # anything
        _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=marginal_bins)
        marginals = np.zeros((true_params.shape[1], marginal_bins))
        for param_n in range(true_params.shape[1]):
            marginals[param_n], _ = np.histogram(galaxy_samples[:, :, param_n], density=True, bins=param_bins)
        f.create_dataset('marginals', data=marginals)


def get_galaxy_save_file(i, save_dir, chain=0):
    return os.path.join(save_dir, f'galaxy_{i}_performance_{chain}.h5')


def get_galaxy_save_file_next_chain(i, save_dir):
    n = 0
    assert os.path.isdir(save_dir)
    while True:
        attempted_save_loc = get_galaxy_save_file(i, save_dir, chain=n)
        if not os.path.isfile(attempted_save_loc):
            return attempted_save_loc, n
        n += 1  # until you find one not yet saved


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
