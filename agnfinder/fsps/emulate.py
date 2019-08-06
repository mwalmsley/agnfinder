import os
import pickle

import h5py
import numpy as np
import GPy


class GPEmulator():

    def __init__(self, gp_model_loc, pca_model_loc):

        with open(pca_model_loc, 'rb') as handle:
            self._pca_model = pickle.load(handle)

        self._gp_model = GPy.models.GPRegression.load_model(gp_model_loc)


    def emulator(self, params):
        """
        Emulates new spectra from physical parameters.
        - params : physical parameters: tau, tage, and dust. 1D vector in that order.

        Output :
        - reconstructed : Emulated image
        """
        # Weights prediction
        params = np.expand_dims(params, axis = 0)
        pred_weights = gp_predict(self._gp_model, params)

        # Inverse PCA (pred_weights * basis + mean)
        reconstructed = self._pca_model.inverse_transform(pred_weights)
        return reconstructed[0]


    def __call__(self, params):
        return self.emulator(params)


def gp_predict(model, params):
    """
    Predicts the weights matrix to feed inverse PCA from physical parameters.

    Input :
    - model : GP model
    - params : physical parameters (flux, radius, shear profile, psf fwhm)

    Output :
    - predic[0] : predicted weights
    """
    predict = model.predict(params)
    return predict[0]


if __name__ == '__main__':

    model_dir = 'notebooks'
    num_params = 3
    num_bases = 10
    gp_model_loc = os.path.join(model_dir, 'gpfit_'+str(num_bases)+'_'+str(num_params) + '.zip')

    pca_model_loc = os.path.join(model_dir, 'pcaModel.pickle')

    save_loc = "/media/mike/internal/agnfinder"
    with h5py.File(os.path.join(save_loc, 'fsps_cache.hdf5'), 'r') as f:
        X_loaded = f['fsps_cache']['X'][...]
        Y_loaded = f['fsps_cache']['Y'][:, 100:]

    ## logging and clipping
    # X_loaded = np.log10(X_loaded)
    Y_log = np.log10(Y_loaded)

    # ### rescaling 
    y_mean = np.mean(Y_log, axis=0)
    y_mult = np.max(Y_log - y_mean, axis=0)

    y_train = (Y_log - y_mean)/y_mult

    # ### rescaling 
    x_mean = np.mean(X_loaded, axis=0)
    x_mult = np.max(X_loaded - x_mean, axis=0)

    x_train = (X_loaded - x_mean)/x_mult

    emulator = GPEmulator(gp_model_loc=gp_model_loc, pca_model_loc=pca_model_loc)
    y = emulator(x_train[0])
    print(y)
