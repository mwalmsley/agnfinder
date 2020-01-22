import os
import argparse
import dill
from collections import OrderedDict

import numpy as np
import pandas as pd

from agnfinder.prospector import main, visualise
from agnfinder import simulation_utils


def simulate(n_samples, save_loc, emulate_ssp, noise, redshift_range):

    # a bit hacky - log* keys will be 10 ** within simulator function below
    free_params = OrderedDict({
        'redshift': [0., 4.],  # the order is really important - redshift is 0th theta index if free. See notebooks/understanding_forward_model
        'log_mass': [8, 12], 
        'dust2': [0., 2.],  # not age of universe, as mistakenly done before...
        'tage': [0.001, 13.8],  # not 0->2, as mistakenly done before...might consider bringing the bounds a bit tighter
        'log_tau': [np.log10(.1), np.log10(30)],  # careful, this is log prior! >2, has very little effect
        'log_agn_mass': [-7, np.log10(15)],  # i.e. from 10**-7 to 15 (not 10**15!)
        'agn_eb_v': [0., 0.5],
        'log_agn_torus_mass': [-7, np.log10(15)]
    })
    param_dim = len(free_params.keys())

    # unit hypercube, n_samples random-ish points
    hcube = simulation_utils.get_unit_latin_hypercube(param_dim, n_samples)
    # tweak redshift normalised theta to lie within desired range
    hcube[:, 0] = simulation_utils.shift_redshift_theta(hcube[:, 0], free_params['redshift'], redshift_range)
    # transform random-ish points back to parameter space (including log if needed)
    galaxy_params = simulation_utils.denormalise_hypercube(hcube, free_params)  
    
    simulator, phot_wavelengths = get_forward_model(emulate_ssp, noise=noise)

    # calculate photometry at every vector (row) in parameter-space 
    photometry = simulation_utils.sample(
        theta=galaxy_params,
        n_samples=n_samples,
        output_dim=12,  # reliable bands only
        simulator=simulator
    )
    print('photometry')
    print(photometry)

    simulation_utils.save_samples(
        save_loc=save_loc,
        theta_names=free_params.keys(),
        theta=galaxy_params,
        normalised_theta=hcube,
        simulator_outputs=photometry,
        wavelengths=phot_wavelengths
    )

def get_forward_model(emulate_ssp, noise):
    # redshift = 3.  # for fixed redshift
    redshift = True
    agn_mass = True
    agn_eb_v = True
    agn_torus_mass = True
    igm_absorbtion = True

    if noise:
        def get_sigma(x):
            return x / 20.
    else:
        get_sigma = None

    _, obs, model, sps = main.construct_problem(redshift=redshift, agn_mass=agn_mass, agn_eb_v=agn_eb_v, agn_torus_mass=agn_torus_mass, igm_absorbtion=igm_absorbtion, emulate_ssp=emulate_ssp)

    _ = visualise.calculate_sed(model, model.theta, obs, sps)  # TODO might not be needed for obs phot wavelengths
    phot_wavelengths = obs['phot_wave']  # always the same, as measured in observer frame
    def forward_model(theta):  # theta must be denormalised!
        assert theta[1] > 1e7  # check mass is properly large
        _, model_photometry, _ = visualise.calculate_sed(model, theta, obs, sps)  # via closure
        if get_sigma is not None:
            return np.random.normal(loc=model_photometry, scale=get_sigma(model_photometry))
        return model_photometry

    return forward_model, phot_wavelengths


if __name__ == '__main__':

    """
    Create a hypercube of (galaxy parameters, photometry) paired vectors, calculated according to the forward model
    Useful for training a neural network to emulate the forward model

    Optionally, use the GP emulator for the forward model. Not a great idea, as this is slower than the original forward model, but I implemented it already...

    Example use: 
        python agnfinder/simulation_samples.py 1000 --z-min 0. --z-max 4. --save-dir data/cubes
    """
    parser = argparse.ArgumentParser(description='Find AGN!')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('--z-min', dest='redshift_min', default=0, type=float)
    parser.add_argument('--z-max', dest='redshift_max', default=4., type=float)
    parser.add_argument('--save-dir', dest='save_dir', type=str, default='data')
    parser.add_argument('--emulate-ssp', default=False, action='store_true')
    parser.add_argument('--noise', default=False, action='store_true')
    args = parser.parse_args()

    redshift_min_string = '{:.4f}'.format(args.redshift_min).replace('.', 'p')
    redshift_max_string = '{:.4f}'.format(args.redshift_max).replace('.', 'p')
    save_name = 'photometry_simulation_{}n_z_{}_to_{}.hdf5'.format(args.n_samples, redshift_min_string, redshift_max_string)
    save_loc = os.path.join(args.save_dir, save_name)

    simulate(args.n_samples, save_loc, args.emulate_ssp, args.noise, (args.redshift_min, args.redshift_max))
