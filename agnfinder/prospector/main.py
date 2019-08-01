import logging
import argparse
import cProfile

import sys
import os
import time
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import corner
import scipy
import matplotlib.pyplot as plt 
import emcee
import dynesty

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from agnfinder.prospector import demo_builders, cpz_builders, visualise, fitting
from agnfinder import columns


def load_galaxy(index=0, galaxy_class=None):  # temp
    try:
        data_dir='/media/mike/internal/agnfinder'
        assert os.path.isdir(data_dir)
    except AssertionError:
        data_dir='data/agnfinder'
        assert os.path.isdir(data_dir)
    logging.info('Using {} as data dir'.format(data_dir))

    parquet_loc = os.path.join(data_dir, 'cpz_paper_sample_week3.parquet')
    cols = columns.cpz_cols['metadata'] + columns.cpz_cols['unified'] + columns.cpz_cols['galex'] + columns.cpz_cols['sdss'] + columns.cpz_cols['cfht'] + columns.cpz_cols['kids'] + columns.cpz_cols['vvv'] + columns.cpz_cols['wise'] + columns.cpz_cols['random_forest']
    df = pd.read_parquet(parquet_loc, columns=cols)
    logging.info('parquet loaded')
    df_with_spectral_z = df[~pd.isnull(df['redshift'])].query('redshift > 1e-2').query('redshift < 4').reset_index()
    if galaxy_class is not None:
        logging.warning('Selecting RF-identified {} galaxies'.format(galaxy_class))
        df_with_spectral_z = df_with_spectral_z.sort_values('Pr[{}]_case_III'.format(galaxy_class), ascending=False).reset_index()  # to pick quasars
    return df_with_spectral_z.iloc[index]


def construct_problem(galaxy, redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion):
    run_params = {}

    # model params
    run_params["object_redshift"] = None
    run_params["fixed_metallicity"] = 0.  # solar
    run_params["add_duste"] = True
    run_params['dust'] = True
    run_params['redshift'] = redshift
    run_params["zcontinuous"] = 1
    run_params['agn_mass'] = agn_mass
    run_params['agn_eb_v'] = agn_eb_v
    run_params['agn_torus_mass'] = agn_torus_mass
    run_params['igm_absorbtion'] = igm_absorbtion

    run_params["verbose"] = False

    logging.info('Run params: {}'.format(run_params))

    obs = cpz_builders.build_cpz_obs(galaxy)
    logging.info(obs)

    # demo_model = demo_builders.build_model(**run_params)
    model = cpz_builders.build_model(**run_params)
    logging.info(model)

    # must come AFTER model?
    sps = cpz_builders.build_sps(**run_params)
    logging.info(sps)

    return run_params, obs, model, sps


def fit_galaxy(run_params, obs, model, sps):

    run_params["dynesty"] = False
    run_params["emcee"] = False
    run_params["optimize"] = True
    run_params["min_method"] = 'lm'
    run_params["nmin"] = 5

    logging.info('Begin minimisation')
    output = fit_model(obs=obs, model=model, sps=sps, lnprobfn=lnprobfn, **run_params)  # careful, modifies model in-place: model['optimization'], model['theta']
    
    time_elapsed = output["optimization"][1]
    log_string = "Done optimization in {}s".format(time_elapsed)
    logging.info(log_string)
    
    logging.info('model theta: {}'.format(model.theta))
    best_theta = fitting.get_best_theta(model, output)

    return best_theta, time_elapsed

def mcmc_galaxy(run_params, obs, model, sps, initial_theta=None, test=False):

    # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
    # https://docs.python.org/3/library/concurrent.futures.html
    # https://monte-python.readthedocs.io/en/latest/nested.html
    # https://johannesbuchner.github.io/PyMultiNest/pymultinest.html

    # Set this to False if you don't want to do another optimization
    # before emcee sampling (but note that the "optimization" entry 
    # in the output dictionary will be (None, 0.) in this case)
    # If set to true then another round of optmization will be performed 
    # before sampling begins and the "optmization" entry of the output
    # will be populated.

    if initial_theta is not None:
        model.set_parameters(initial_theta)
        logging.info('Setting initial params: {}'.format(dict(zip(model.free_params, initial_theta))))

    if test:
        logging.warning('Running emcee in test mode')
        nwalkers = 16
        niter = 8
        nburn = [16]
    else:
        nwalkers = 128
        niter = 256
        nburn = [16]

    run_params["optimize"] = True  # find MLE first
    run_params["emcee"] = True
    run_params["dynesty"] = False
    # Number of emcee walkers
    run_params["nwalkers"] = nwalkers
    # Number of iterations of the MCMC sampling
    run_params["niter"] = niter  # was 512
    # Number of iterations in each round of burn-in
    # After each round, the walkers are reinitialized based on the 
    # locations of the highest probablity half of the walkers.
    run_params["nburn"] = nburn

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    sampler = output['sampling'][0]
    time_elapsed = output["sampling"][1]
    logging.info('done emcee in {0}s'.format(time_elapsed))

    samples = sampler.flatchain
    return samples, time_elapsed


def dynesty_galaxy(run_params, obs, model, sps, initial_theta=None, test=False):

    run_params["dynesty"] = True
    run_params["optimize"] = False
    run_params["emcee"] = False

    dynesty_params = {}
    dynesty_params["nested_method"] = "rwalk"
    dynesty_params["nlive_init"] = 400
    dynesty_params["nlive_batch"] = 200
    dynesty_params["nested_dlogz_init"] = 0.05
    dynesty_params["nested_posterior_thresh"] = 0.05
    dynesty_params["nested_maxcall"] = int(1e7)
    logging.info('Running dynesty with params {}'.format(dynesty_params))
    run_params.update(dynesty_params)

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    samples = output["sampling"][0].samples
    time_elapsed = output["sampling"][1]
    log_string = 'done dynesty in {0}s'.format(time_elapsed)
    logging.info(log_string)

    return samples, time_elapsed

    # def log_likelihood(theta):
    #     # print(theta, 'theta internal')
    #     # assert len(theta) > 1
    #     pbar.update()
    #     return lnprobfn(theta, model=model, sps=sps, obs=obs)

    # sampler = emcee.EnsembleSampler(
    #     nwalkers,
    #     ndim,
    #     log_likelihood,
    #     pool=None,
    #     # backend=backend
    # )

    # # start every walker at the same point
    # initial_position = np.array(list(initial_theta) * nwalkers).reshape(nwalkers, ndim)
    # print('initial_position', initial_position.shape)
    # start_time = datetime.now()
    # sampler.run_mcmc(initial_position, nsteps)
    # end_time = datetime.now()
    # time_elapsed = end_time - start_time


def save_samples(samples, model, file_loc):
    with h5py.File(file_loc, "w") as f:
        dset = f.create_dataset('samples', samples.shape, dtype='float32')
        dset[...] = samples


def save_corner(samples, model, file_loc):
    figure = corner.corner(samples, labels=model.free_params,
                        show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(file_loc)


def save_sed_traces(samples, obs, model, sps, file_loc, max_samples=1000, burn_in=1000):
    # Let's exclude the first 1000, just in case. Only really needed for nested sampling. Need to revew the algorithm details here!
    model_dim = len(model.free_params)
    sample_dim = len(samples[0])
    if not model_dim == sample_dim:
        raise ValueError(
            'check model of with params {} ({}) matches the one used for samples ({}) - free redshift? free AGN?'.format(model.free_params, model_dim, sample_dim)
        )
    if len(samples) < max_samples:
        sample_indices = range(burn_in, len(samples))  # display all
    else:  # pick random subset of samples to display
        n_samples = min(len(samples - burn_in), max_samples)  # 1000 samples, or all remaining
        sample_indices = np.random.choice(range(burn_in, len(samples)), size=n_samples, replace=False)  
    visualise.visualise_obs_and_models(obs, model, samples[sample_indices], sps)
    plt.savefig(file_loc)


def main(index, name, galaxy_class, redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion, find_ml_estimate, find_mcmc_posterior, find_multinest_posterior):

    galaxy = load_galaxy(index, galaxy_class)
    logging.info('Galaxy: {}'.format(galaxy))
    logging.info('with spectro. redshift: {}'.format(galaxy['redshift']))

    if redshift == 'spectro':
        redshift = galaxy['redshift']
    run_params, obs, model, sps = construct_problem(
        galaxy,
        redshift=redshift,
        agn_mass=agn_mass,
        agn_eb_v=agn_eb_v,
        agn_torus_mass=agn_torus_mass,
        igm_absorbtion=igm_absorbtion
    )

    if find_ml_estimate:
        theta_best, _ = fit_galaxy(run_params, obs, model, sps)
        logging.info(list(zip(model.free_params, theta_best)))
        # TODO save best_theta to json?
        visualise.visualise_obs_and_model(obs, model, theta_best, sps)
        plt.savefig(os.path.join(output_dir, '{}_ml_estimate.png'.format(name)))
        plt.clf()
    else:
        theta_best = None

    if find_mcmc_posterior:
        samples, _ = mcmc_galaxy(run_params, obs, model, sps, initial_theta=theta_best, test=test)
        sample_loc = os.path.join(output_dir, '{}_mcmc_samples.h5py'.format(name))
        save_samples(samples, model, sample_loc)
        corner_loc = os.path.join(output_dir, '{}_mcmc_corner.png'.format(name))
        save_corner(samples, model, corner_loc)
        traces_loc = os.path.join(output_dir, '{}_mcmc_sed_traces.png'.format(name))
        save_sed_traces(samples[int(len(samples)/2):], obs, model, sps, traces_loc)

    if find_multinest_posterior:
        # TODO extend to use pymultinest
        samples, _ = dynesty_galaxy(run_params, obs, model, sps, initial_theta=theta_best, test=test)
        sample_loc = os.path.join(output_dir, '{}_multinest_samples.h5py'.format(name))
        save_samples(samples, model, sample_loc)
        corner_loc = os.path.join(output_dir, '{}_multinest_corner.png'.format(name))
        save_corner(samples[int(len(samples)/2):], model, corner_loc)  # nested sampling has no burn-in phase, early samples are bad
        traces_loc = os.path.join(output_dir, '{}_multinest_sed_traces.png'.format(name))
        save_sed_traces(samples, obs, model, sps, traces_loc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('index', type=int, help='index of galaxy to fit')
    parser.add_argument('galaxy', type=str, help='class of galaxy to fit')
    parser.add_argument('--profile', type=bool, default=False, dest='profile')
    args = parser.parse_args()

    timestamp = '{:.0f}'.format(time.time())
    # TODO convert to command line args?
    name = '{}_loguniform_mass_{}_{}'.format(args.galaxy, args.index, timestamp)
    output_dir = 'results'
    find_ml_estimate = False
    find_mcmc_posterior = False
    find_multinest_posterior = True
    test = False
    redshift = 'spectro'  # exception to below, as redshift read from galaxy
    agn_mass = True  # None for not modelled, True for free, or float for fixed
    agn_eb_v = True
    agn_torus_mass = True
    igm_absorbtion = True
    
    galaxy_class = args.galaxy # None for any, or 'agn', 'passive', 'starforming', 'qso' for most likely galaxies of that class

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(
        filename=os.path.join(output_dir, '{}.log'.format(name)),
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    logging.debug('Script ready')

    if args.profile:
        logging.warning('Using profiling')
        pr = cProfile.Profile()
        pr.enable()
    main(args.index, name, galaxy_class, redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion, find_ml_estimate, find_mcmc_posterior, find_multinest_posterior)
    if args.profile:
        pr.disable()
        pr.dump_stats(os.path.join(output_dir, '{}.profile'.format(name)))
