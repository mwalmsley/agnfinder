import logging

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

logging.info('Loading complete')

# re-defining plotting defaults
# from matplotlib.font_manager import FontProperties
# from matplotlib import gridspec
# plt.rcParams.update({'xtick.major.pad': '7.0'})
# plt.rcParams.update({'xtick.major.size': '7.5'})
# plt.rcParams.update({'xtick.major.width': '1.5'})
# plt.rcParams.update({'xtick.minor.pad': '7.0'})
# plt.rcParams.update({'xtick.minor.size': '3.5'})
# plt.rcParams.update({'xtick.minor.width': '1.0'})
# plt.rcParams.update({'ytick.major.pad': '7.0'})
# plt.rcParams.update({'ytick.major.size': '7.5'})
# plt.rcParams.update({'ytick.major.width': '1.5'})
# plt.rcParams.update({'ytick.minor.pad': '7.0'})
# plt.rcParams.update({'ytick.minor.size': '3.5'})
# plt.rcParams.update({'ytick.minor.width': '1.0'})
# plt.rcParams.update({'xtick.color': 'k'})
# plt.rcParams.update({'ytick.color': 'k'})
# plt.rcParams.update({'font.size': 30})
# logging.info('Matplotlib update complete')

def load_galaxy():  # temp
    try:
        data_dir='/media/mike/internal/agnfinder'
        assert os.path.isdir(data_dir)
    except AssertionError:
        data_dir='data/agnfinder'
        assert os.path.isdir(data_dir)
    logging.info('Using {} as data dir'.format(data_dir))

    parquet_loc = os.path.join(data_dir, 'cpz_paper_sample.parquet')
    cols = columns.cpz_cols['metadata'] + columns.cpz_cols['unified'] + columns.cpz_cols['galex'] + columns.cpz_cols['sdss'] + columns.cpz_cols['cfht'] + columns.cpz_cols['kids'] + columns.cpz_cols['vvv'] + columns.cpz_cols['wise']
    df = pd.read_parquet(parquet_loc, columns=cols)
    logging.info('parquet loaded')
    return df.iloc[0]


def construct_problem(galaxy, redshift):
    run_params = {}

    # obs params
    run_params["snr"] = 10.0

    # model params
    run_params["object_redshift"] = None
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = True
    run_params['dust'] = True
    run_params['redshift'] = redshift
    run_params["zcontinuous"] = 1

    run_params["verbose"] = True

    logging.debug('Run params defined')

    obs = cpz_builders.build_cpz_obs(galaxy, snr=10.)
    logging.debug('obs built')

    sps = demo_builders.build_sps(**run_params)
    logging.debug('sps built')

    # demo_model = demo_builders.build_model(**run_params)
    model = cpz_builders.build_model(**run_params)
    logging.debug('model built')
    # if run_params['object_redshift'] is None:
    #     assert len(demo_model.initial_theta) < len(model.initial_theta)
    # else:
    #     assert all(demo_model.initial_theta == model.initial_theta)


    return run_params, obs, model, sps


def fit_galaxy(run_params, obs, model, sps):

    # --- start minimization ----
    run_params["dynesty"] = False
    run_params["emcee"] = False
    run_params["optimize"] = True
    run_params["min_method"] = 'lm'
    run_params["nmin"] = 5

    # model.initial_theta
    logging.warning('Begin minimisation')
    output = fit_model(obs=obs, model=model, sps=sps, lnprobfn=lnprobfn, **run_params)  # careful, modifies model in-place: model['optimization'], model['theta']
    
    time_elapsed = output["optimization"][1]
    log_string = "Done optimization in {}s".format(time_elapsed)
    logging.warning(log_string)
    print(log_string)
    
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

    # model = model.copy()
    if initial_theta is not None:
        model.set_parameters(initial_theta)
        

    # ndim = len(model.theta)
    # ndim = len(initial_theta)

    if test:
        logging.warning('Running emcee in test mode')
        nwalkers = 16
        niter = 256
        nburn = [16]  # about 2.5 minutes, nearly all from initialising FSPS model (judging from very slow first likelihood eval)
        # nsteps = 1000
    else:
        nwalkers = 128
        niter = 64   # 128 * 16 walkers took about 5 minutes, 64 walkers about 15ish
        nburn = [16, 32, 64]
        # nsteps = 10000

    print(initial_theta)
    # print(ndim)

    run_params["optimize"] = False
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

    # pool = Pool()
    pool = None
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params, pool=pool)
    print('done emcee in {0}s'.format(output["sampling"][1]))

    sampler = output['sampling'][0]
    time_elapsed = output["sampling"][1]
    
    # backend = emcee.backends.HDFBackend('temp.hdf5')
    # backend.reset(nwalkers, ndim)
    # # avoid using lambda as can't be pickled -> can't be threaded
    # pbar = tqdm(total=nsteps * nwalkers, unit='steps')
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

    return sampler, time_elapsed


if __name__ == '__main__':

    # TODO convert to command line args
    name = 'custom_model_free_redshift.png'
    output_dir = 'results'
    find_ml_estimate = True
    find_mcmc_posterior = True
    test = True
    redshift = 0.08  # now using fixed redshift to check degeneracies

    logging.basicConfig(
        filename=os.path.join(output_dir, '{}.log'.format(name)),
        format='%(asctime)s %(message)s',
        level=logging.DEBUG)

    logging.debug('Script ready')

    timestamp = '{:.0f}'.format(time.time())
    
    galaxy = load_galaxy()

    run_params, obs, model, sps = construct_problem(galaxy, redshift=redshift)

    if find_ml_estimate:
        theta_best, time_elapsed = fit_galaxy(run_params, obs, model, sps)
        # TODO save best_theta to json?
        visualise.visualise_obs_and_model(obs, model, theta_best, sps)
        plt.savefig(os.path.join(output_dir, '{}_{}_ml_estimate.png'.format(name, timestamp)))
        plt.clf()
    else:
        theta_best = None

    # temporary hardcoded
    # theta_best = [
    #      2.16027863e-01,  6.16566688e+06, -1.74889834e+00,  3.49055535e-01, 2.32118967e+00,  3.72065557e-01
    # ]

    # theta_best = [
    #     3.86485209e-03,  9.83688011e+06, -7.49336715e-01,  4.97978128e-01, 6.49960994e+00,  1.96756967e+00
    # ]

    if find_mcmc_posterior:
        sampler, mcmc_time_elapsed = mcmc_galaxy(run_params, obs, model, sps, initial_theta=theta_best, test=test)


    # with h5py.File('{}_{}.hdf5'.format(name, timestamp), "w") as f:
    #     dset = f.create_dataset(name, sampler.flatchain.shape, dtype='float32')
    #     dset[...] = sampler.flatchain
        # hfile = "demo_emcee_mcmc_v2.h5"
        # writer.write_hdf5(hfile, run_params, model, obs, sampler, tsample=mcmc_time_elapsed)

    figure = corner.corner(sampler.flatchain, labels=model.free_params,
                        show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(output_dir, '{}_{}_corner.png'.format(name, timestamp)))
