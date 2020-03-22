import logging
import argparse
import cProfile
import os
import time
import datetime

import h5py
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model

from agnfinder.prospector import cpz_builders, visualise, fitting, load_photometry
from agnfinder import columns, simulation_samples, simulation_utils
from agnfinder.tf_sampling import run_sampler, deep_emulator


def load_catalog(catalog_loc):
    # catalog_loc could be '../cpz_paper_sample_week3_maggies.parquet'
    # assume catalog has already had mag and maggie columns calculated
    # can do this with data exploration notebook that creates the parquet
    logging.info('Using {} as catalog'.format(catalog_loc))

    filters = load_photometry.get_filters(selection='euclid')
    required_cols = [f.maggie_col for f in filters] + [f.maggie_error_col for f in filters] + ['redshift'] + columns.cpz_cols['metadata'] + columns.cpz_cols['random_forest']

    if catalog_loc.endswith('.parquet'):
        df = pd.read_parquet(catalog_loc, columns=required_cols)
    else:
        df = pd.read_csv(catalog_loc, usecols=required_cols)  # why, pandas, is this a different named arg?

    df = df.dropna(subset=required_cols)

    df_with_spectral_z = df[~pd.isnull(df['redshift'])].query('redshift > 1e-2').query('redshift < 4').reset_index()
    return df_with_spectral_z


def load_galaxy(catalog_loc, index=0, forest_class=None, spectro_class=None):
    df = load_catalog(catalog_loc)  # will filter to galaxies with z only - see above
    if forest_class is not None:
        logging.warning('Selecting forest-identified {} galaxies'.format(forest_class))
        df = df.sort_values('Pr[{}]_case_III'.format(forest_class), ascending=False).reset_index(drop=True)  # to pick quasars
    if spectro_class is not None:
        logging.warning('Selecting spectro-identified {} galaxies'.format(spectro_class))
        df = df.query('hclass == {}'.format(spectro_class)).reset_index()  # to pick quasars
    return df.reset_index(drop=True).iloc[index]


def construct_problem(redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion, inclination, emulate_ssp, filter_selection, galaxy=None):
    run_params = {}

    # model params
    # these get passed to build_model and build_sps
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
    run_params['inclination'] = inclination
    run_params['emulate_ssp'] = emulate_ssp

    run_params["verbose"] = False

    logging.info('Run params: {}'.format(run_params))

    if galaxy is None:
        logging.warning('Galaxy not supplied - creating default obs for Prospector compatability only')
    obs = cpz_builders.build_cpz_obs(galaxy=galaxy, filter_selection=filter_selection)
    logging.info(obs)

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
    run_params["nmin"] = 50

    logging.info('Begin minimisation')
    output = fit_model(obs=obs, model=model, sps=sps, lnprobfn=lnprobfn, **run_params)  # careful, modifies model in-place: model['optimization'], model['theta']
    
    results = output["optimization"][0]
    time_elapsed = output["optimization"][1]
    log_string = "Done optimization in {}s".format(time_elapsed)
    logging.info(log_string)
    
    logging.info('model theta: {}'.format(model.theta))
    best_theta = fitting.get_best_theta(model, output)

    return best_theta, results, time_elapsed

def mcmc_galaxy(run_params, obs, model, sps, initial_theta=None, test=False):


    # https://github.com/bd-j/prospector/blob/bb5deaed0140887665079761efba657a11f876af/prospect/fitting/ensemble.py
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
        # BE CAREFUL that these match the suggested values in run_sampler_singlethreaded.py, for a fair comparison
        # nwalkers = 32  # ndim=8 * walker_factor=4, prospector defaults
        # nwalkers = 16
        nwalkers = 128
        # niter = 256
        # niter = 10000  # i.e. iterations of emcee, somewhat like steps  # 1 hour w/ 32 chains, x10 for 256 chains
        niter = 10000 # long run to check convergence
        # niter = 500  # short, to check everything works
        # niter = 2086 * 2
        nburn = [5000]

    logging.info(f'Walkers: {nwalkers}. Iterations: {niter}. Burnin: {nburn}')

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

    # can't parallelise as involves a lambda, and probably would mess with the fortran driver
    # is a parallel problem, of course

    logging.info(f'Starting emcee at {datetime.datetime.now()}')
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)

    sampler = output['sampling'][0]
    time_elapsed = output["sampling"][1]
    logging.info('done emcee in {:.1f}s'.format(time_elapsed))

    samples = sampler.flatchain
    # samples = sampler.get_chain(flat=False)
    return samples, time_elapsed


def dynesty_galaxy(run_params, obs, model, sps, test=False):
    # test not implemented

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


def save_samples(samples, model, obs, sps, file_loc, name, true_theta=None):
    if 'zred' in model.fixed_params: # other fixed params irrelevant
        fixed_param_names = ['redshift']
        fixed_params = [model.params['zred']]
    else:
        fixed_param_names = []
        fixed_params = []
    run_sampler.save_galaxy(
        save_file=file_loc,
        galaxy_samples=samples,   # will include meaningful chain dim.
        galaxy_n=0,
        free_param_names=model.free_params,
        init_method='NA',
        n_burnin=0,
        name=name,
        attempt_n=0,
        sample_weights=0,
        log_evidence=0,
        true_observation=obs['maggies'],
        fixed_params=fixed_params,
        fixed_param_names=fixed_param_names,
        uncertainty=obs['maggies_unc'],
        metadata={},
        true_params=true_theta
    )


def save_corner(samples, model, file_loc):
    if samples.ndim > 2:
        samples = samples.reshape(-1, samples.shape[2])  # flatten chains
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


def main(index, name, catalog_loc, save_dir, forest_class, spectro_class, redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion, inclination, find_ml_estimate, find_mcmc_posterior, find_multinest_posterior, emulate_ssp, cube_loc):

    if catalog_loc == '':
        assert args.cube_loc is not ''

        # load randomly from cube
        # _, _, x_test, y_test = deep_emulator.data(cube_dir=cube_loc)  # TODO apply selection filters
        # OR load from preselected test sample
        x_test = np.loadtxt('data/cubes/x_test_v2.npy')
        y_test = np.loadtxt('data/cubes/y_test_v2.npy')

        assert y_test.shape[1] == 8  # euclid cube
        cube_photometry = deep_emulator.denormalise_photometry(y_test[index])  # other selection params have no effect
        cube_params = x_test[index]  # only need for redshift
        # pretend cube is catalog photometry
        filters = load_photometry.get_filters('euclid')
        galaxy = {}
        for n, f in enumerate(filters):
            galaxy[f.maggie_col] = cube_photometry[n]
        galaxy['redshift'] = cube_params[0] * 4.  # 4 to scale from hcube. Again, crucial not to change param_lims!
        true_theta = cube_params[1:]  # fixed redshift, normalised. Need to denormalise at end, or (better?) normalise samples.
        logging.info(f'True theta: {true_theta}')
        
        # estimate_maggie_unc expects a batch dimension, so temporarily add one
        maggie_uncertainty = np.squeeze(load_photometry.estimate_maggie_uncertainty(np.expand_dims(cube_photometry, axis=0)))
        for n, f in enumerate(filters):
            galaxy[f.maggie_error_col] = maggie_uncertainty[n]
    else:
        galaxy = load_galaxy(catalog_loc, index, forest_class, spectro_class)
        true_theta = None

    logging.info('Galaxy: {}'.format(galaxy))
    logging.info('with spectro. redshift: {}'.format(galaxy['redshift']))

    if redshift == 'spectro':
        redshift = galaxy['redshift']

    run_params, obs, model, sps = construct_problem(
        galaxy=galaxy,  # now a kwarg that's none by default, as we usually only want the forward model
        redshift=redshift,
        agn_mass=agn_mass,
        agn_eb_v=agn_eb_v,
        agn_torus_mass=agn_torus_mass,
        igm_absorbtion=igm_absorbtion,
        inclination=inclination,
        emulate_ssp=emulate_ssp,
        filter_selection='euclid'
    )

    start_time = datetime.datetime.now()
    logging.info('Beginning inference (not counting model construction) at {}'.format(start_time))

    if find_ml_estimate:
        theta_best, results, _ = fit_galaxy(run_params, obs, model, sps)
        assert np.any([r.cost < 10**4 for r in results])  # by eye, anything worse than this is 'no good start found' and should exit
        logging.info(list(zip(model.free_params, theta_best)))
        # TODO save best_theta to json?
        visualise.visualise_obs_and_model(obs, model, theta_best, sps)
        plt.savefig(os.path.join(save_dir, '{}_ml_estimate.png'.format(name)))
        plt.clf()
    else:
        theta_best = None

    if find_mcmc_posterior:
        samples, _ = mcmc_galaxy(run_params, obs, model, sps, initial_theta=theta_best, test=test)
        limits_without_redshift = simulation_samples.FREE_PARAMS.copy()
        del limits_without_redshift['redshift']
        normalised_samples = simulation_utils.normalise_theta(samples, limits_without_redshift)
        normalised_samples = np.expand_dims(normalised_samples, axis=1)  # add chain dim back in
        sample_loc = os.path.join(save_dir, '{}_mcmc_samples.h5'.format(name))
        save_samples(normalised_samples, model, obs, sps, sample_loc, name, true_theta=true_theta)
        corner_loc = os.path.join(save_dir, '{}_mcmc_corner.png'.format(name))
        save_corner(normalised_samples, model, corner_loc)
        # traces_loc = os.path.join(save_dir, '{}_mcmc_sed_traces.png'.format(name))
        # save_sed_traces(samples[int(len(samples)/2):], obs, model, sps, traces_loc)

    if find_multinest_posterior:
        # TODO extend to use pymultinest?
        samples, _ = dynesty_galaxy(run_params, obs, model, sps, test=test)
        sample_loc = os.path.join(save_dir, '{}_multinest_samples.h5'.format(name))
        save_samples(samples, model, obs, sps, sample_loc, name, true_theta=true_theta)
        corner_loc = os.path.join(save_dir, '{}_multinest_corner.png'.format(name))
        save_corner(samples[int(len(samples)/2):], model, corner_loc)  # nested sampling has no burn-in phase, early samples are bad
        # traces_loc = os.path.join(save_dir, '{}_multinest_sed_traces.png'.format(name))
        # save_sed_traces(samples, obs, model, sps, traces_loc)
        # components_loc = os.path.join(save_dir, '{}_multinest_components.png'.format(name))
        # messy saving of components
        # plt.clf()
        # visualise.calculate_many_components(model, samples[int(len(samples)/2):], obs, sps)
        # plt.legend()
        # plt.ylim([1e-25, None])
        # plt.tight_layout()
        # plt.savefig(components_loc)


if __name__ == '__main__':

    """
    Fit a forward model to a real galaxy.
    Provide a catalog index to select which galaxy. Useful for running in parallel.
    Optionally, filter to only one class of galaxy (labelled by spectrosopy or random forests)
    
    Output: samples (.h5py) and corner plot of forward model parameter posteriors for the selected galaxy

    Example use:
    python agnfinder/prospector/main.py cube_test --cube data/cubes/latest --save-dir results/vanilla_mcmc
    python agnfinder/prospector/main.py passive --catalog-loc data/uk_ir_selection_577.parquet --save-dir results/vanilla_nested --forest passive
    """
    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('name', type=str, help='name of run')
    parser.add_argument('--catalog', dest='catalog_loc', type=str, default='')
    parser.add_argument('--cube', dest='cube_loc', type=str, default='')
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--index', type=int, default=0, dest='index', help='index of galaxy to fit')
    parser.add_argument('--forest', type=str, default=None, dest='forest', help='forest-estimated class of galaxy to fit')
    parser.add_argument('--spectro', type=str, default='any', dest='spectro', help='filter to only galaxies with this spectro. label, before selecting by index')
    parser.add_argument('--profile', default=False, dest='profile', action='store_true')
    parser.add_argument('--emulate-ssp', default=False, dest='emulate_ssp', action='store_true')
    parser.add_argument('--test', default=False, dest='test', action='store_true')
    args = parser.parse_args()

    timestamp = '{:.0f}'.format(time.time())
    name = '{}_{}_{}'.format(args.name, args.index, timestamp)
    save_dir = args.save_dir

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(
        # filename=os.path.join(save_dir, '{}.log'.format(name)),
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    find_ml_estimate = True  # do both!
    find_mcmc_posterior = True
    find_multinest_posterior = False
    test = args.test
    redshift = 'spectro'  # exception to below, as redshift read from galaxy
    igm_absorbtion = True

    # None for not modelled, True for free, or float for fixed
    agn_mass = True  
    agn_eb_v = True
    agn_torus_mass = True
    inclination = True
    # agn_mass = None
    # agn_eb_v = None
    # agn_torus_mass = None
    # inclination = None

    if args.catalog_loc == '':  # use the cube
        assert args.cube_loc
        forest_class = None
        spectro_class = None
    else:  # use a real galaxy
        # None or 'random' for any, or agn', 'passive', 'starforming', 'qso' for most likely galaxies of that class
        if args.forest == 'random':
            forest_class = None
        else:
            forest_class = args.forest
        hclass_schema = {
            'galaxy': 1,
            'agn': 2,
            'qso': 3,
            'any': None
        }
        spectro_class = hclass_schema[args.spectro]

    if args.profile:
        logging.warning('Using profiling')
        pr = cProfile.Profile()
        pr.enable()
    main(args.index, name, args.catalog_loc, save_dir, forest_class, spectro_class, redshift, agn_mass, agn_eb_v, agn_torus_mass, igm_absorbtion, inclination, find_ml_estimate, find_mcmc_posterior, find_multinest_posterior, args.emulate_ssp, args.cube_loc)
    if args.profile:
        pr.disable()
        pr.dump_stats(os.path.join(save_dir, '{}.profile'.format(name)))
