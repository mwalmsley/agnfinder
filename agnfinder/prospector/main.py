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
from scipy import optimize

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model

from agnfinder.prospector import cpz_builders, visualise, fitting, load_photometry
from agnfinder import columns, simulation_samples, simulation_utils
from agnfinder.tf_sampling import run_sampler, deep_emulator, api, emcee_sampling
from prospect.fitting import minimizer


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
    run_params["nmin"] = 1
    run_params['min_opts'] = {
        # 'xtol': 1e-10,
        # 'ftol': 1,
        'verbose': 1
    }

    logging.info('Begin minimisation')
    output = fit_model(obs=obs, model=model, sps=sps, lnprobfn=lnprobfn, **run_params)  # careful, modifies model in-place: model['optimization'], model['theta']
    # print(output['optimization'])
    # print([x for x in output['optimization']])
    # print('cost: ', output['optimization']['cost'])

    results = output["optimization"][0]
    time_elapsed = output["optimization"][1]
    log_string = "Done optimization in {}s".format(time_elapsed)
    logging.info(log_string)
    
    logging.info('model theta: {}'.format(model.theta))
    best_theta = fitting.get_best_theta(model, output)

    return best_theta, results, time_elapsed


def fit_galaxy_manual(model, obs, sps):

    fsps_forward_model = simulation_samples.wrap_fsps_model(model, obs, sps)
    # for this, I only want the least squares flux deviation
    # I want to pass in normalised params including redshfit, plus scale,
    # denormalise them in the forward model step
    # log-multiply by scale
    # and denormalise them again once we have the best theta

    free_params_no_z = simulation_samples.FREE_PARAMS
    del free_params_no_z['redshift'] # unusually, z is fixed

    def forward_model(x):
        params = x[:-1]
        scale = x[-1]
        # params = x  # pass scale by closure, no need to optimise?
        assert len(params) == 8
        # denormalise params, remembering there's no redshift
        params_denormed = simulation_utils.denormalise_theta(params.reshape(1, -1), free_params_no_z).squeeze()
        fsps_photometry = fsps_forward_model(params_denormed) 
        norm_photometry = deep_emulator.normalise_photometry(fsps_photometry.reshape(1, -1), rescale=True)
        photometry = deep_emulator.denormalise_photometry(norm_photometry, scale)
        return photometry.squeeze()

    def residuals(x):
        predicted_flux = forward_model(x)
        # return predicted_flux - obs['maggies']
        return np.abs(np.log10(predicted_flux) - np.log10(obs['maggies']))

    initial = model.theta.copy()
    initial_params = minimizer.minimizer_ball(initial, 1, model)  # just get one, seems to just pick the init though
    initial_params = np.squeeze(initial_params)  # initial_params must be 1D
    initial_params_normed = simulation_utils.normalise_theta(initial_params.reshape(1, -1), free_params_no_z).squeeze()
    initial_scale = -np.log10(obs['maggies']).sum().reshape(1)  # 1D. Should really refactor this...
    x0 = np.concatenate([initial_params_normed, initial_scale], axis=-1)

    logging.info('Begin minimisation')
    bounds = ([0. for _ in initial_params] + [0], [1. for _ in initial_params] + [1e4])
    results = optimize.least_squares(residuals, x0, bounds=bounds)
    best_theta_normed = results.x
    # back to real theta, for consistency
    best_params = simulation_utils.denormalise_theta(best_theta_normed[:-1].reshape(1, -1), free_params_no_z).squeeze()
    best_scale = best_theta_normed[-1]
    best_theta = np.concatenate([best_params, best_scale.reshape(1)], axis=-1)
    return best_theta, results


def mcmc_galaxy_manual(model, obs, sps, theta):  # requires previous minimisation

    # nwalkers = 32  # for now, 256 on Zeus w/ GPU
    # n_burnin = 5000
    # n_samples = 20000  # up from 10k, can cut artifically later

    nwalkers = 32  # for now, 256 on Zeus w/ GPU
    n_burnin = 5000
    n_samples = 20000  # up from 10k, can cut artifically later

    fsps_forward_model = simulation_samples.wrap_fsps_model(model, obs, sps)
    free_params_no_z = simulation_samples.FREE_PARAMS
    if 'redshift' in free_params_no_z.keys():
        del free_params_no_z['redshift'] # unusually, z is fixed

    def forward_model(x, param_dim=8):  # expects batch
        if x.ndim == 1:
            x = x.reshape(1, -1)
        params = x[:, :-1]
        scale = x[:, -1]
        assert params.shape[1] == param_dim  # remembering scale
        params = np.clip(params, 1e-5, 1-1e-5)  # log prob will make these be rejected, but the forward model failure messes up the shapes

        # denormalise params, remembering there's no redshift
        params_denormed = simulation_utils.denormalise_theta(params, free_params_no_z)
        # fsps can't handle batch, calculate sequentially
        fsps_predictions = [fsps_forward_model(x_row) for x_row in params_denormed]
        shapes = [p.shape for p in fsps_predictions]
        try:
            fsps_predictions = np.stack(fsps_predictions, axis=0)
        except ValueError:
            print(params_denormed)
            print(shapes)
            print(fsps_predictions)
            raise ValueError

        normalised_predictions = deep_emulator.normalise_photometry(fsps_predictions, rescale=True)
        predictions = deep_emulator.denormalise_photometry(normalised_predictions, scale.reshape(-1, 1))
        return predictions


    def log_prob_fn(x):  # expects variable batch size
        if x.ndim == 1:
            batch_dim = 1
        else:
            batch_dim = x.shape[0]
        true_flux = np.tile(obs['maggies'].reshape(1, -1), [batch_dim, 1])
        uncertainty = np.tile(obs['maggies_unc'].reshape(1, -1), [batch_dim, 1])
        # no scale param
        predicted_flux = forward_model(x)
        deviation = np.abs(predicted_flux - true_flux)   # make sure you denormalise true observation in the first place, if loading from data(). Should be in maggies.
        variance = uncertainty ** 2
        neg_log_prob_by_band = 0.5*( (deviation**2/variance) - np.log(2*np.pi*variance) )  # natural log, not log10
        log_prob = -neg_log_prob_by_band.sum(axis=1)  # log space: product -> sum
        x_out_of_bounds = api.is_out_of_bounds_python(x[:, :-1])  # being careful to exclude scale
        penalty = x_out_of_bounds * 1e10
        log_prob_with_penalty = log_prob - penalty  # no effect if x in bounds, else divide (subtract) a big penalty
        log_prob_with_penalty[log_prob_with_penalty < -1e9] = -np.inf
        return log_prob_with_penalty

    # test only
    initial_params = theta[:-1]
    norm_params = simulation_utils.normalise_theta(initial_params.reshape(1, -1), free_params_no_z).squeeze()
    print('norm theta: ', norm_params)
    # norm_theta = theta
    # log_prob = log_prob_fn(norm_theta.reshape(1, -1))
    # print(log_prob)

    logging.info('Begin manual emcee')
    x0_ball = emcee_sampling.get_initial_starts_ball(theta.reshape(1, -1), neg_log_p=np.ones(1), nwalkers=nwalkers)  # only one found, if it's reliable stick w/ this
    samples, _ = emcee_sampling.run_emcee(nwalkers, x0_ball, n_burnin, n_samples, log_prob_fn)
    return samples
    # return samples.reshape(-1, len(theta))  # flatten for historical checks, for now


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
        nwalkers = 128
         # i.e. iterations of emcee, somewhat like steps  # 1 hour w/ 32 chains, x10 for 256 chains
        niter = 10000 # long run to check convergence
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

    samples = sampler.flatchain  # may change
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
        fixed_params = [model.params['zred'] / 4.]  # should be div 4, to be normalised consistently w/ other routines
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
    if samples.shape[-1] == 8:
        labels = model.free_params
    if samples.shape[-1] == 9:
        labels = list(model.free_params) + ['scale'] 
    else:
        labels = None
    figure = corner.corner(samples, labels=labels,
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
        # cube loc doesn't do anything any more, we just load directly from the test set

        # load randomly from cube
        # _, _, x_test, y_test = deep_emulator.data(cube_dir=cube_loc)  # TODO apply selection filters
        # OR load from preselected test sample
        x_test = np.loadtxt('data/cubes/x_test_latest.npy')
        y_test = np.loadtxt('data/cubes/y_test_latest.npy')
        assert y_test.shape[1] == 8  # euclid cube

        cube_params = x_test[index]  # only need for redshift

        galaxy = {}
        galaxy['redshift'] = cube_params[0] * 4.  # denormalised, 4 to scale from hcube. Again, crucial not to change param_lims!

        scale = y_test[index].sum()  # already neglog10 normalised
        true_theta = np.concatenate(cube_params[1:], scale.reshape(1))  # fixed redshift, normalised, with scale added
        logging.info(f'True theta: {true_theta}')

        filters = load_photometry.get_filters('euclid')

        cube_photometry = deep_emulator.denormalise_photometry(y_test[index], scale=1.)  # other selection params have no effect WARNING SCALE

        # estimate_maggie_unc expects a batch dimension, so temporarily add one
        maggie_uncertainty = np.squeeze(load_photometry.estimate_maggie_uncertainty(np.expand_dims(cube_photometry, axis=0)))

        # make a mock observation
        observed_photometry = np.random.normal(loc=cube_photometry, scale=maggie_uncertainty)
        # pretend observed photometry is in catalog
        for n, f in enumerate(filters):
            galaxy[f.maggie_col] = observed_photometry[n]
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
        theta_best, results = fit_galaxy_manual(model, obs, sps)
        # theta_best, results, _ = fit_galaxy(run_params, obs, model, sps)
        # assert np.any([r.cost < 10**4 for r in results])  # by eye, anything worse than this is 'no good start found' and should exit
        print(results.cost)
        assert results.cost < 1
        logging.info(list(zip(model.free_params, theta_best[:-1])))
        # TODO save best_theta to json?
        visualise.visualise_obs_and_model(obs, model, theta_best[:-1], sps)
        plt.savefig(os.path.join(save_dir, '{}_ml_estimate.png'.format(name)))
        plt.clf()
    else:
        theta_best = None

    # theta_best = np.array([0.69517508, 0.78529271, 0.02695847, 0.09131194, 0.88317726, 0.4672737, 0.70804482, 0.15671718])

    if find_mcmc_posterior:
        assert theta_best is not None

        manual = True
        if manual:
            normalised_samples = mcmc_galaxy_manual(model, obs, sps, theta_best)  # normalised params, with chain dimension
        else:
            unnormalised_samples = mcmc_galaxy(run_params, obs, model, sps, initial_theta=theta_best, test=test)  # fsps params, with no chain dimension
            limits_without_redshift = simulation_samples.FREE_PARAMS.copy()
            if 'redshift' in limits_without_redshift.keys():
                del limits_without_redshift['redshift']
            normalised_samples = simulation_utils.normalise_theta(unnormalised_samples, limits_without_redshift)
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
    python agnfinder/prospector/main.py galaxy --cube data/cubes/latest --save-dir results/vanilla_emcee
    python agnfinder/prospector/main.py passive --catalog-loc data/uk_ir_selection_577.parquet --save-dir results/vanilla_nested --forest passive
    """
    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('--name', type=str, help='name of run', dest='name', default='galaxy')
    parser.add_argument('--catalog', dest='catalog_loc', type=str, default='')
    parser.add_argument('--cube', dest='cube_loc', type=str, default='data/cubes/latest')
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='results/vanilla_emcee_local')
    parser.add_argument('--index', type=int, default=8, dest='index', help='index of galaxy to fit')
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
