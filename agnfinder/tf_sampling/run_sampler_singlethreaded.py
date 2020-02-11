import argparse
import logging
import os

import dill
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf  # just for eager toggle
import statsmodels.api as sm
from scipy.interpolate import interp1d
import h5py

from agnfinder.prospector import load_photometry
from agnfinder.tf_sampling import run_sampler, deep_emulator, parameter_recovery


# TODO will change to some kind of unique id for each galaxy, rather than the index
def get_galaxies_to_run(n_galaxies):
    indices_to_run = []
    i = 0
    while len(indices_to_run) < n_galaxies:
        if not is_galaxy_successful(save_dir, i):
            indices_to_run.append(i)
        i += 1
    return indices_to_run


def is_galaxy_successful(save_dir, galaxy_n):
    chain_n = 0
    while True:
        file_loc = run_sampler.get_galaxy_save_file(galaxy_n, save_dir, chain=chain_n)
        if os.path.isfile(file_loc):  # check if it succeeded
            if run_succeeded(file_loc):
                return True
            else:
                chain_n += 1
        else:  # not a file, not yet attempted
            return False


def run_succeeded(file_loc):
    # some overlap with parameter_recovery.py
    with h5py.File(file_loc, mode='r') as f:
        samples = f['samples'][...] # okay to load, will not keep
        within_limits = parameter_recovery.within_percentile_limits(samples)  # WARNING limits will need updating for new cubes/uncertainties!
        logging.info(within_limits)
        return within_limits


def record_performance_on_galaxies(checkpoint_loc, selected_catalog_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir, fixed_redshift, filter_selection):
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    n_photometry = 12

    always_free_param_names = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling', 'inclination']
    if fixed_redshift:
        fixed_param_names = ['redshift']
        free_param_names = always_free_param_names
    else:
        fixed_param_names = []
        free_param_names = ['redshift'] + always_free_param_names
    logging.info('Free params: {}'.format(free_param_names))
    logging.info('Fixed params: {}'.format(fixed_param_names))

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if selected_catalog_loc is not '':
        # real galaxies, selected from uK_IR sample by highest (?) random forest prob. (`Pr[{class}]_case_III')
        assert os.path.isfile(selected_catalog_loc)  # TODO generalise?
        df = pd.read_parquet(selected_catalog_loc)
        n_galaxies = np.min([len(df), n_chains])
        df = df.sample(n_galaxies)  # don't reset index, important for labels
        logging.info(f'Loading {len(df)} galaxies')
        rf_classes = ['passive', 'starforming', 'starburst', 'agn', 'qso', 'outlier']
        for c in rf_classes:
            logging.info('{}: {:.2f}'.format(c, df[f'Pr[{c}]_case_III'].sum()))
        true_observation = np.zeros((len(df), n_photometry)).astype(np.float32)  # fixed bands
        uncertainty = np.zeros_like(true_observation).astype(np.float32)
        if fixed_redshift:
            fixed_params = np.zeros((len(df), 1), dtype=np.float32)  # will hold redshifts
        else:
            fixed_params = np.zeros((len(df), 0), dtype=np.float32)  # note the 0 shape
        for n in tqdm(range(len(df))):
            galaxy = df.iloc[n]
            _, maggies, maggies_unc = load_photometry.load_maggies_from_galaxy(galaxy, filter_selection)
            uncertainty[n] = maggies_unc.astype(np.float32)  # trusting the catalog uncertainty, which may be brave
            true_observation[n] = maggies.astype(np.float32)
            if fixed_redshift:
                logging.info('Using fixed spectroscopic redshifts from catalog')
                # div by 4 to convert to hypercube space
                # TODO WARNING assumes cube max redshift is 4, absolutely must match cube redshift limits and prior is uniform
                fixed_params[n] = galaxy['redshift'] / 4. 
        true_params = np.zeros((len(df), len(free_param_names))).astype(np.float32)  # easier than None as I often use it in asserts or for expected param dim
        logging.warning(f'Using {len(df)} real galaxies - forcing n_chains from {n_chains} to {len(df)} accordingly')
        n_chains = len(df)  # overriding whatever the arg was
        galaxy_indices = df.index  # just in case the df index was not reset to 0...n

    else:
        # fake galaxies, drawn from our priors and used as emulator training data
        logging.info('Using fake galaxies, drawn randomly from the hypercube')
        _, _, x_test, y_test = deep_emulator.data(cube_dir='data/cubes/latest')  # TODO could make as arg
        # filter to max redshift .5
        within_max_z = x_test[:, 0] < .5 / 4.

        # even with 100x factor, only 25% of galaxies don't get uncertainty clipped
        # many galaxies probably still unrealistic
        # should make more histograms / read them to find good clips
        # should maybe have two cubes: low-z cube and any/high-z cube?
        # photometry on low-z galaxies can be a 'targeted special case' 
        # already have the code to do it
        # run zeus tonight?
        # min_maggies = 10 ** -8
        # max_maggies = 10 ** -6
        # normalised_min_maggies = deep_emulator.normalise_photometry(min_maggies)
        # normalised_max_maggies = deep_emulator.normalise_photometry(max_maggies)
        # within_maggie_limit = np.all((y_test < normalised_min_maggies) & (y_test > normalised_max_maggies), axis=1)  # less than, as normalising includes a sign flip

        astro_acceptable = within_max_z
        # astro_acceptable = within_maggie_limit
        # astro_acceptable = within_max_z & within_maggie_limit
        x_test = x_test[astro_acceptable]
        y_test = y_test[astro_acceptable]
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        galaxy_indices = get_galaxies_to_run(n_chains)
        # galaxy_indices = np.arange(n_chains)   # if re-run, is effectively a new chain for an old galaxy
        print(galaxy_indices)
        exit()

        if fixed_redshift:
            logging.info('Using fixed redshifts from cube')
            true_params = x_test[galaxy_indices, 1:]  # excluding the 0th redshift param, which we treat as fixed
            fixed_params = x_test[galaxy_indices, :1].astype(np.float32)  # shape (n_galaxies, 1)
        else:
            logging.info('Using variable redshift')
            true_params = x_test[galaxy_indices]
            fixed_params = np.zeros((len(true_params), 0)).astype(np.float32)
        true_observation = deep_emulator.denormalise_photometry(y_test[galaxy_indices]) 
        assert filter_selection == 'euclid'
        bands = ['u_sloan', 'g_sloan', 'r_sloan', 'i_sloan', 'z_sloan', 'VISTA_H','VISTA_J', 'VISTA_Y']  # euclid bands, hardcoded for now
        assert true_observation.shape[1] == len(bands)
        # lowess = sm.nonparametric.lowess
        error_estimators_loc = 'data/error_estimators.pickle'
        with open(error_estimators_loc, 'rb') as f:
            error_estimators = dill.load(f)
        estimated_uncertainty = np.zeros_like(true_observation).astype(np.float32)
        for galaxy_i, galaxy in enumerate(true_observation):
            for band_i, band in enumerate(bands):
                estimated_uncertainty[galaxy_i, band_i] = error_estimators[band](galaxy[band_i])
        # add clipping
        uncertainty = np.min(np.stack([estimated_uncertainty, true_observation * 0.15]), axis=0)  # 1 sigma uncertainty no more than 15%
        uncertainty = np.max(np.stack([uncertainty, true_observation * 0.03]), axis=0)  # no less than 3%
#         uncertainty = true_observation * 0.05  # assume 5% uncertainty on all bands for simulated galaxies

    logging.info('photometry: ')
    logging.info(true_observation)
    logging.info('Uncertainty: ')
    logging.info(uncertainty)
    logging.info('Uncertainty (decimal)')
    logging.info(uncertainty / true_observation)
    logging.info('Mean uncertainty by band (decimal):')
    logging.info(np.mean(uncertainty / true_observation, axis=0))
    
    assert len(fixed_params) == len(true_observation) == len(true_params)
    run_sampler.sample_galaxy_batch(
        galaxy_indices,
        true_observation,
        fixed_params,
        uncertainty,
        true_params,
        emulator,
        n_burnin,
        n_samples,
        n_chains,
        init_method,
        save_dir,
        free_param_names,
        fixed_param_names
    )


if __name__ == '__main__':

    """
    Run the emulated HMC method on many galaxies, in a single thread.
    Evaluating performance at recovering posteriors can be done in `evaluate_performance.py`

    Example use: 
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling --n-chains 4 --n-samples 100 --n-burnin 100 --init random
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling --selected data/uk_ir_selection_577.parquet

    Default burn-in, num samples, and num chains are optimised for an excellent desktop w/ GTX 1070. 
    """
    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc', default='results/checkpoints/latest')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='results/emulated_sampling')  # in which save_dir will be created
    parser.add_argument('--max-galaxies', type=int, default=1, dest='max_galaxies')
    parser.add_argument('--selected', type=str, default='', dest='selected_catalog_loc')
    parser.add_argument('--n-burnin', type=int, default=3000, dest='n_burnin')
    parser.add_argument('--n-samples', type=int, default=80000, dest='n_samples')
    parser.add_argument('--n-chains', type=int, default=512, dest='n_chains')
    parser.add_argument('--init', type=str, dest='init_method', default='optimised', help='Can be one of: random, roughly_correct, optimised')
    parser.add_argument('--redshift', type=str, dest='redshift_str', default='fixed', help='Can be one of: fixed, free')
    parser.add_argument('--filters', type=str, dest='filters', default='euclid', help='Can be one of: euclid, reliable. Only has an effect on real galaxies.')
    args = parser.parse_args()
    
    if args.redshift_str == 'fixed':
        fixed_redshift = True
    elif args.redshift_str == 'free':
        fixed_redshift = False
    else:
        raise ValueError('Redshift {} not understood - should be "fixed" or "free'.format(args.redshift_str))
    
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    checkpoint_loc =  args.checkpoint_loc
    output_dir = args.output_dir
    assert checkpoint_loc is not None
    assert output_dir is not None
    max_galaxies = args.max_galaxies
    n_burnin = args.n_burnin
    n_samples = args.n_samples
    n_chains = args.n_chains
    init_method = args.init_method
    selected_catalog_loc = args.selected_catalog_loc

    if selected_catalog_loc is not '':
        logging.info('Using real galaxies from {}'.format(selected_catalog_loc))
        save_dir = os.path.join(output_dir, os.path.basename(selected_catalog_loc)).split('.')[0]  # no periods otherwise allowed...
    else:
        logging.info('Using simulated galaxies')
        save_dir = os.path.join(output_dir, 'latest_{}_{}_{}'.format(n_samples, n_chains, init_method))
        # save_dir = 'results/emulated_sampling/30k_burnin'

    logging.info('Galaxies will save to {}'.format(save_dir))
    record_performance_on_galaxies(checkpoint_loc, selected_catalog_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir, fixed_redshift, args.filters)
    # run_sampler.aggregate_performance(save_dir, n_samples, chains_per_galaxy=1)
    # samples, true_params, true_observations = run_sampler.read_performance(save_dir)
    # print(samples.shape, true_params.shape, true_observations.shape)
