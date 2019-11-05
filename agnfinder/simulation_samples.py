import os
import argparse
import dill
from collections import OrderedDict

import numpy as np
import pandas as pd

from agnfinder.prospector import main, visualise
from agnfinder import simulation_utils

def simulate(n_samples, catalog_loc, save_loc, emulate_ssp, noise):

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

    simulator, phot_wavelengths = get_photometry_simulator(catalog_loc, emulate_ssp, noise=noise)

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

def get_photometry_simulator(catalog_loc, emulate_ssp, noise):
    galaxy_index = 1
    galaxy = main.load_galaxy(catalog_loc, galaxy_index)
    redshift = galaxy['redshift']
    agn_mass = True
    agn_eb_v = True
    agn_torus_mass = True
    igm_absorbtion = True

    if noise:
        def get_sigma(x):
            return x / 20.
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

    """
    Create a hypercube of (galaxy parameters, photometry) paired vectors, calculated according to the forward model
    Useful for training a neural network to emulate the forward model

    Optionally, use the GP emulator for the forward model. Not a great idea, as this is slower than the original forward model, but I implemented it already...

    Example use: /data/miniconda3/envs/agnfinder/bin/python /Data/repos/agnfinder/agnfinder/simulation_samples.py 100 --catalog-loc /Volumes/alpha/agnfinder/cpz_paper_sample_week3.parquet --save-dir data
    """
    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('--catalog-loc', dest='catalog_loc', type=str)
    parser.add_argument('--save-dir', dest='save_dir', type=str)
    parser.add_argument('--emulate-ssp', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    args = parser.parse_args()

    save_name = 'photometry_simulation_{}.hdf5'.format(args.n_samples)
    save_loc = os.path.join(args.save_dir, save_name)

    simulate(args.n_samples, args.catalog_loc, save_loc, args.emulate_ssp, args.noise)
