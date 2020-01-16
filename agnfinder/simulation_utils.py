import logging

import tqdm
import numpy as np
import h5py
import pyDOE2

def shift_redshift_theta(norm_redshift, fixed_theta_range, target_theta_range):
    # adjust spread in norm space
    norm_redshift = norm_redshift  * (target_theta_range[1] - target_theta_range[0]) / (fixed_theta_range[1] - fixed_theta_range[0])
    # adjust start in norm space
    return norm_redshift + target_theta_range[0] / fixed_theta_range[1]
    
def get_unit_latin_hypercube(dims, n_samples):
    return pyDOE2.lhs(n=dims, samples=n_samples, criterion='correlation')


def denormalise_hypercube(normalised_hcube, limits):
    # normalised_theta_to_sample.shape
    theta_to_sample = normalised_hcube.copy()
    for key_n, (key, lims) in enumerate(limits.items()):
        # 0, 1 -> -2, 6
        # 0, 8
        theta_to_sample[:, key_n] = (theta_to_sample[:, key_n] * (lims[1] - lims[0]) + lims[0]) 
        # print(key, theta_to_sample[:, key_n].min(), theta_to_sample[:, key_n].max())
        if key.startswith('log'):
            logging.info('Automatically exponentiating {}'.format(key))
            theta_to_sample[:, key_n] = 10 ** theta_to_sample[:, key_n]
    return theta_to_sample


def denormalise_theta(normalised_theta, limits):
    theta = np.zeros_like(normalised_theta)
    for n, (key, lims) in enumerate(limits.items()):
        theta[n] = normalised_theta[n] * (lims[1] - lims[0]) + lims[0]
        if key.startswith('log'):
            theta[n] = 10 ** theta[n]
    return theta


def sample(theta, n_samples, output_dim, simulator):
    """
    Calculate simulator(theta) for every vector (row) in theta
    

    Args:
        theta ([type]): [description]
        n_samples ([type]): [description]
        output_dim ([type]): [description]
        simulator ([type]): [description]
    
    Returns:
        X: array of simulator input parameters
        Y: array of simulator output vectors, matching X index (and *I think* theta_df index)
    """
    Y = np.zeros((n_samples, output_dim))  # simulator output vectors
    for n in tqdm.tqdm(range(len(theta))):
        Y[n] = simulator(theta[n])
    return Y


def save_samples(save_loc, theta_names, theta, normalised_theta, simulator_outputs, wavelengths=None):
    with h5py.File(save_loc, 'w') as f:
        grp = f.create_group('samples')
        ds_x = grp.create_dataset('theta', data=theta)
        ds_x.attrs['columns'] = list(theta_names)
        ds_x.attrs['description'] = 'Parameters used by simulator'
        
        ds_x_norm = grp.create_dataset('normalised_theta', data=normalised_theta)
        ds_x_norm.attrs['description'] = 'Normalised parameters used by simulator'

        ds_y = grp.create_dataset('simulated_y', data=simulator_outputs)
        ds_y.attrs['description'] = 'Response of simulator'
        
        # hacky
        if wavelengths is not None:
            ds_wavelengths = grp.create_dataset('wavelengths', data=wavelengths)
            ds_wavelengths.attrs['description'] = 'Observer wavelengths to visualise simulator photometry'
    logging.info(f'Saved samples to {save_loc}')
