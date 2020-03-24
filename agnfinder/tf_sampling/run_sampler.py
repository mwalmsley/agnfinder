import os
import logging
import argparse
import json  # temporary, for debugging

from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf  # just for eager toggle

from agnfinder.tf_sampling import deep_emulator, hmc, nested, emcee_sampling

# TODO change indices to some kind of unique id, perhaps? will need for real galaxies...
def sample_galaxy_batch(problem, mode, n_burnin, n_samples, init_method, save_dir, free_param_names, fixed_param_names):

    if mode == 'hmc':
        sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, init_method=init_method)
    # OR
    # nested sampling (currently deprecated)
    # sampler = nested.SamplerNested(problem, n_live=400)   (outputs need updating for consistency)
    elif mode == 'emcee':
        sampler = emcee_sampling.SamplerEmcee(problem, n_burnin, n_samples, init_method=init_method)
    else:
        raise ValueError(f'Mode {mode} not understood')

    successful_ids, samples, sample_weights, log_evidence, metadata = sampler()  # lists, indexed by galaxy

    for galaxy_n, name in tqdm(enumerate(successful_ids), unit=' galaxies saved'):
        save_file, attempt_n = get_galaxy_save_file_next_attempt(name, save_dir)  # save each run with the same id to a new file
        
        # tediously select the outputs relevent to that particular galaxy
        galaxy_samples = samples[galaxy_n]
        assert galaxy_samples.ndim == 3  # (sample, chain, param) dimensions
        galaxy_sample_weights = sample_weights[galaxy_n]
        galaxy_log_evidence = log_evidence[galaxy_n]
        galaxy_true_observation = problem.true_observation[galaxy_n]
        galaxy_fixed_params = problem.fixed_params[galaxy_n]
        galaxy_true_params = problem.true_params[galaxy_n]
        galaxy_metadata = metadata[galaxy_n]
        galaxy_uncertainty = problem.uncertainty[galaxy_n]
    
        save_galaxy(save_file, galaxy_samples, galaxy_n, free_param_names, init_method, n_burnin, name, attempt_n, galaxy_sample_weights, galaxy_log_evidence, galaxy_true_observation, galaxy_fixed_params, fixed_param_names, galaxy_uncertainty, galaxy_metadata, galaxy_true_params)


def save_galaxy(save_file, galaxy_samples, galaxy_n, free_param_names, init_method, n_burnin, name, attempt_n, sample_weights, log_evidence, true_observation, fixed_params, fixed_param_names, uncertainty, metadata, true_params):
    f = h5py.File(save_file, mode='w')  # will overwrite
    # for 0-1 decimals, float16 is more than precise enough and much smaller files
    # scaleoffset=5 means keep only the first 5 decimal places (plenty on 0-1 interval) to save space
    dset = f.create_dataset('samples', data=galaxy_samples, scaleoffset=5)
    dset.attrs['free_param_names'] = free_param_names
    dset.attrs['init_method'] = init_method
    dset.attrs['n_burnin'] = n_burnin
    dset.attrs['galaxy_id'] = name
    f.create_dataset('attempt', data=attempt_n)
    f.create_dataset('sample_weights', data=sample_weights)
    f.create_dataset('log_evidence', data=log_evidence)
    f.create_dataset('true_observations', data=true_observation)
    dset = f.create_dataset('fixed_params', data=fixed_params)
    dset.attrs['fixed_param_names'] = fixed_param_names
    f.create_dataset('uncertainty', data=uncertainty)
    for key, data in metadata.items():
        f.create_dataset(key, data=data)
    if true_params is not None:
        f.create_dataset('true_params', data=true_params)
    # add marginals
    marginal_bins = 50
    dummy_array = np.zeros(42)  # anything
    _, param_bins = np.histogram(dummy_array, range=(0., 1.), bins=marginal_bins)

    print(true_params.shape)
    print(galaxy_samples.shape)
    marginals = np.zeros((len(true_params), marginal_bins))
    for param_n in range(len(true_params)):
        marginals[param_n], _ = np.histogram(galaxy_samples[:, :, param_n], density=True, bins=param_bins)  # galaxy samples is still dim3, confusingly
    f.create_dataset('marginals', data=marginals)


def get_galaxy_save_file(name, save_dir, chain=0):
    return os.path.join(save_dir, f'galaxy_{name}_performance_{chain}.h5')


def get_galaxy_save_file_next_attempt(name, save_dir):
    n = 0
    assert os.path.isdir(save_dir)
    while True:
        attempted_save_loc = get_galaxy_save_file(name, save_dir, chain=n)
        if not os.path.isfile(attempted_save_loc):
            return attempted_save_loc, n
        n += 1  # until you find one not yet saved


# def aggregate_performance(save_dir, n_samples, chains_per_galaxy):
#     logging.info(f'Aggregating galaxies in {save_dir}')
#     assert chains_per_galaxy == 1  # TODO remove as arg?
#     logging.debug('Creating virtual dataset')
#     performance_files = [os.path.join(save_dir, x) for x in os.listdir(save_dir) if x.endswith('_performance.h5')]
#     n_sources = len(performance_files)
#     logging.info('Using source files: {} (max 10 shown)'.format(performance_files[:10]))
#     logging.debug('Specifing expected data')
#     samples_vl = h5py.VirtualLayout(shape=(n_sources, n_samples, chains_per_galaxy, 7), dtype='f')
#     true_params_vl = h5py.VirtualLayout(shape=(n_sources, 7), dtype='f')
#     true_observations_vl = h5py.VirtualLayout(shape=(n_sources, 12), dtype='f')

#     samples_source_shape = (n_samples, chains_per_galaxy, 7)
#     logging.info('shape of samples expected: {}'.format(samples_source_shape))

#     logging.debug('Specifying sources')
#     for i, file_loc in enumerate(performance_files):
#         assert os.path.isfile(file_loc)

#         samples_vl[i] = h5py.VirtualSource(file_loc, 'samples', shape=samples_source_shape)
#         true_params_vl[i] = h5py.VirtualSource(file_loc, 'true_params', shape=(7,))
#         true_observations_vl[i] = h5py.VirtualSource(file_loc, 'true_observations', shape=(12,))

#     # Add virtual dataset to output file
#     logging.debug('Writing virtual dataset')
#     with h5py.File(aggregate_filename(save_dir), 'w') as f:
#         f.create_virtual_dataset('samples', samples_vl, fillvalue=0)
#         f.create_virtual_dataset('true_params', true_params_vl, fillvalue=0)
#         f.create_virtual_dataset('true_observations', true_observations_vl, fillvalue=0)
    

# def read_performance(save_dir):
#     # if the code hangs while reading, there's a shape mismatch between sources and virtual layout - probably samples.
#     file_loc = aggregate_filename(save_dir)
#     return read_h5(file_loc)

# def read_h5(file_loc):
#     with h5py.File(file_loc, 'r') as f:
#         logging.debug('Reading {}'.format(file_loc))
#         logging.debug(list(f.keys()))
#         samples = f['samples'][...]
#         logging.debug('Samples read')
#         true_params = f['true_params'][...]
#         true_observations = f['true_observations'][...]
#     logging.debug('{} loaded'.format(file_loc))
#     return samples, true_params, true_observations


# def aggregate_filename(save_dir):
#     return os.path.join(save_dir, 'all_virtual.h5')
