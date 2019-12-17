import os
import logging
import argparse
import json  # temporary, for debugging

from tqdm import tqdm
import numpy as np
import h5py
import tensorflow as tf  # just for eager toggle
os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'

from agnfinder.tf_sampling import deep_emulator, api, hmc


def run_on_single_galaxy(name, true_observation, true_params, emulator, n_burnin, n_samples, n_chains, init_method, save_dir):

    logging.warning('True params: {}'.format(true_params))
    logging.warning('True observation: {}'.format(true_observation))

    problem = api.SamplingProblem(true_observation, true_params, forward_model=emulator)
    sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
    samples = sampler()

    # TODO this should never actually happen now as directory is named according to shape
    # explicitly remove old files to avoid shape mismatch issues
    save_file = get_galaxy_save_file(name, save_dir)
    if os.path.isfile(save_file):
        os.remove(save_file)

    expected_shape = (n_samples, n_chains, 7)
    if samples.shape != expected_shape:
        logging.warning('Samples not required shape - skipping save to avoid virtual dataset issues')
        logging.warning('actual {} vs expected {}'.format(samples.shape, expected_shape))
    else:
        
        f = h5py.File(save_file, mode='w')
        logging.warning('shape of samples: {}'.format(samples.shape))
        f.create_dataset('samples', data=samples)
        f.create_dataset('true_params', data=true_params)
        f.create_dataset('true_observations', data=true_observation)


def get_galaxy_save_file(i, save_dir):
    return os.path.join(save_dir, 'galaxy_{}_performance.h5'.format(i))


def aggregate_performance(save_dir, n_samples, n_chains):
    logging.debug('Creating virtual dataset')
    performance_files = [os.path.join(save_dir, x) for x in os.listdir(save_dir) if x.endswith('_performance.h5')]
    n_sources = len(performance_files)
    logging.warning('Using source files: {}'.format(performance_files))
    logging.debug('Specifing expected data')
    samples_vl = h5py.VirtualLayout(shape=(n_sources, n_samples, n_chains, 7), dtype='f')
    true_params_vl = h5py.VirtualLayout(shape=(n_sources, 7), dtype='f')
    true_observations_vl = h5py.VirtualLayout(shape=(n_sources, 12), dtype='f')

    logging.debug('Specifying sources')
    for i, file_loc in enumerate(performance_files):
        assert os.path.isfile(file_loc)
        samples_source_shape = (n_samples, n_chains, 7)
        logging.warning('shape of samples expected: {}'.format(samples_source_shape))
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
