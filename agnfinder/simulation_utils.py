import logging

from tqdm import tqdm
import numpy as np
import h5py
import pyDOE2


def get_unit_latin_hypercube(dims, samples):
    return pyDOE2.lhs(n=dims, samples=samples, criterion='correlation')


def denormalise_hypercube(normalised_hcube, limits):
    # normalised_theta_to_sample.shape
    theta_to_sample = normalised_hcube.copy()
    for key_n, (key, lims) in enumerate(limits.items()):
        theta_to_sample[:, key_n] = (theta_to_sample[:, key_n] + lims[0]) *(lims[1] - lims[0])
        if key.startswith('log'):
            logging.warning('Automatically exponentiating {}'.format(key))
            theta_to_sample[:, key_n] = theta_to_sample[:, key_n] ** 10
    return theta_to_sample


def sample(theta_df, n_samples, output_dim, simulator):
    theta_names = theta_df.columns.values  # df with columns (theta_1, theta_2, ...)
    X = np.zeros((n_samples, len(theta_names)))
    Y = np.zeros((n_samples, output_dim))
    for n, theta_tuple in tqdm(enumerate(theta_df.sample(n_samples).itertuples(name='theta'))):
        X[n] = [getattr(theta_tuple, p) for p in theta_names]
        Y[n] = simulator(X[n])
    return X, Y

def save_samples(save_loc, free_params, theta_df, normalised_theta_df, simulated_y, wavelengths=None):
    with h5py.File(save_loc, 'w') as f:
        grp = f.create_group('samples')
        ds_x = grp.create_dataset('theta', data=theta_df.values)
        ds_x.attrs['columns'] = list(theta_df.columns.values)
        ds_x.attrs['description'] = 'Parameters used by simulator'
        
        ds_x_norm = grp.create_dataset('normalised_theta', data=normalised_theta_df.values)
        ds_x_norm.attrs['description'] = 'Normalised parameters used by simulator'

        ds_y = grp.create_dataset('simulated_y', data=simulated_y)
        ds_y.attrs['description'] = 'Response of simulator'
        
        # hacky
        if wavelengths is not None:
            ds_wavelengths = grp.create_dataset('wavelengths', data=wavelengths)
            ds_wavelengths.attrs['description'] = 'Observer wavelengths to visualise simulator photometry'
