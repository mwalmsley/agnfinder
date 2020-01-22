import logging

import tqdm
import numpy as np
import h5py
import pyDOE2


def shift_redshift_theta(norm_redshift, fixed_theta_range, target_theta_range):
    """Take the normalised redshift for this cube, which initally be 0->1,
    and transform so that when used with other cubes, they will all cover the 0->1 range when combined (not each)
    
    As a consequence, when denomalised, the transformed redshift will cover the target redshift range *for this cube* 
    
    Args:
        norm_redshift (np.array): initial normalised redshift, from 0 to 1 (as with other normalised theta)
        fixed_theta_range (tuple): min/max (astrophysical) redshift *across all cubes*
        target_theta_range (tuple): min/max (astrophysical) redshift *for this cube*
    
    Returns:
        np.array: Transformed normalised redshift, e.g. array covering 0.4->0.6, corresponding to z=1.2->1.8
    """
    # adjust spread in norm space
    norm_redshift = norm_redshift  * (target_theta_range[1] - target_theta_range[0]) / (fixed_theta_range[1] - fixed_theta_range[0])
    # adjust start in norm space
    return norm_redshift + target_theta_range[0] / fixed_theta_range[1]
    
def get_unit_latin_hypercube(dims, n_samples):
    return pyDOE2.lhs(n=dims, samples=n_samples, criterion='correlation')


def denormalise_hypercube(hcube, limits):
    assert hcube.shape[0] == len(limits.keys())
    return np.transpose(denormalise_theta(hcube.tranpose()))


def denormalise_theta(normalised_theta, limits):
    assert normalised_theta.shape[1] == len(limits.keys())
    # convert hypercube to real parameter space
    # expects dim1 to have which param, dim0 to have param values
    # equivalent to denormalise_theta after a transpose - TODO combine
    physical_theta = normalised_theta.copy()
    for key_n, (key, lims) in enumerate(limits.items()):
        physical_theta[:, key_n] = denormalise_param(physical_theta[:, key_n], lims,  key.startswith('log'))
        # print(key, theta_to_sample[:, key_n].min(), theta_to_sample[:, key_n].max())
    return physical_theta


# def denormalise_theta(normalised_theta, limits):
#     # convert arbitrary rows of normalised parameters to real parameter space
#     # expects normalised_theta to have dim0=which param, dim1=param values
#     theta = np.zeros_like(normalised_theta)
#     for n, (key, lims) in enumerate(limits.items()):
#         theta[n] = denormalise_param(normalised_theta[n], lims, key.startswith('log'))
#     return theta


def denormalise_param(normalised_param, param_limits, log):
        physical_range = param_limits[1] - param_limits[0]
        stretched = normalised_param * physical_range  # full normalised range is always 0->1
        shifted = stretched + param_limits[0]
        if log:
            return 10 ** np.clip(shifted, -10, 20)  # stay within 10^-10 -> 10^20 
        return shifted
        

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
