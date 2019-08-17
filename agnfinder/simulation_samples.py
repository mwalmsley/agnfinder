import os
import argparse
import dill
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prospect.utils.obsutils import fix_obs
import pyDOE2

from agnfinder.prospector import main, cpz_builders, visualise
from agnfinder import simulation_utils

def simulate(n_samples, save_loc, emulate_ssp, noise):

    # a bit hacky - log* keys will be 10 ** within simulator function below
    free_params = OrderedDict({
        'log_mass': [8, 12], 
        'dust2': [0.001, 13.8],
        'tage': [0., 2.],
        'tau': [.1, 30],
        'log_agn_mass': [-7, np.log10(15)],
        'agn_eb_v': [0., 0.5],
        'log_agn_torus_mass': [-7, np.log10(15)]
    })
    param_dim = len(free_params.keys())

    hcube = simulation_utils.get_unit_latin_hypercube(param_dim, n_samples)
    denormalised_hcube = simulation_utils.denormalise_hypercube(hcube, free_params)
    
    theta_df = pd.DataFrame(denormalised_hcube, columns=free_params.keys())
    normalised_theta_df = pd.DataFrame(hcube, columns=free_params.keys())

    simulator, phot_wavelengths = get_photometry_simulator(emulate_ssp, noise=noise)

    # not using the normalised params from simulation_utils, sticking with normalised_theta_df. Very messy TODO
    _, photometry = simulation_utils.sample(
        theta_df=theta_df,
        n_samples=n_samples,
        output_dim=12,  # reliable bands only
        simulator=simulator
    )

    simulation_utils.save_samples(
        save_loc=save_loc,
        free_params=free_params,
        theta_df=theta_df,
        normalised_theta_df=normalised_theta_df,
        simulated_y=photometry,
        wavelengths=phot_wavelengths
    )

def get_photometry_simulator(emulate_ssp, noise):
    galaxy_index = 1
    galaxy = main.load_galaxy(galaxy_index)
    redshift = galaxy['redshift']
    agn_mass = True
    agn_eb_v = True
    agn_torus_mass = True
    igm_absorbtion = True
    emulate_ssp = emulate_ssp

    if noise:
        # with open('data/error_estimators.pickle', 'rb') as f:
            # error_estimators = dill.load(f)  # get_sigma needs this via closure
        def get_sigma(x):
            # best_guess = np.array([error_estimators[band](x[n]) for n, band in enumerate(error_estimators.keys())])
            # if np.isnan(best_guess).any():
            return x / 20.
            # else:
                # return best_guess
    else:
        get_sigma = None 

    _, obs, model, sps = main.construct_problem(galaxy, redshift=redshift, agn_mass=agn_mass, agn_eb_v=agn_eb_v, agn_torus_mass=agn_torus_mass, igm_absorbtion=igm_absorbtion, emulate_ssp=emulate_ssp)

    _ = visualise.calculate_sed(model, model.theta, obs, sps)  # TODO might not be needed for obs phot wavelengths
    phot_wavelengths = obs['phot_wave']
    def simulator(theta):  # theta must be denormalised!
        assert theta[0] > 1e7  # check mass is properly large
        _, model_photometry, _ = visualise.calculate_sed(model, theta, obs, sps)  # via closure
    #     phot_wavelengths = obs['phot_wave']  # always the same, as in observer frame
        if get_sigma is not None:
            return np.random.normal(loc=model_photometry, scale=get_sigma(model_photometry))
        return model_photometry

    return simulator, phot_wavelengths


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('--emulate-ssp', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    # parser.add_argument('save_loc', type=str, dest='save_loc')
    args = parser.parse_args()

    # n_samples = 100
    save_loc = 'data/photometry_simulation_{}.hdf5'.format(args.n_samples)

    simulate(args.n_samples, save_loc, args.emulate_ssp, args.noise)
