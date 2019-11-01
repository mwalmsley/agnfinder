import os
import pickle

import h5py
import numpy as np
import GPy
import matplotlib.pyplot as plt


class SKLearnEmulator():

    def __init__(self, model_loc):
        with open(model_loc, 'rb') as f:
            self._model = pickle.load(f)

    def __call__(self, params):
        return self._model.predict(params.reshape(1, -1))


class GPEmulator():

    def __init__(self, gp_model_loc, pca_model_loc):

        # assume these files are in same dir as gp_model_loc
        model_loc = os.path.dirname(gp_model_loc)
        self.x_mean = np.loadtxt(os.path.join(model_loc, 'x_mean.txt'))
        self.x_mult = np.loadtxt(os.path.join(model_loc, 'x_mult.txt'))
        self.y_mean = np.loadtxt(os.path.join(model_loc, 'y_mean.txt'))
        self.y_mult = np.loadtxt(os.path.join(model_loc, 'y_mult.txt'))

        with open(pca_model_loc, 'rb') as handle:
            self._pca_model = pickle.load(handle)

        self._gp_model = GPy.models.GPRegression.load_model(gp_model_loc)


    def emulator(self, params):
        """
        Emulates new spectra from physical parameters.
        - params : physical parameters: tau, tage, and dust. 1D vector in that order.

        Output :
        - reconstructed : Emulated target (Y)
        """
        # normalise params
        params = (params - self.x_mean)/self.x_mult
    
        # Weights prediction
        params = np.expand_dims(params, axis = 0)
        pred_weights = gp_predict(self._gp_model, params)

        # Inverse PCA (pred_weights * basis + mean)
        reconstructed = self._pca_model.inverse_transform(pred_weights)
        # denormalise
        return 10**((reconstructed[0]*self.y_mult) + self.y_mean)


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
    return model.predict(params)[0]


def test_gp(params):
    model_dir = 'notebooks'
    num_params = 3
    num_bases = 10
    gp_model_loc = os.path.join(model_dir, 'gpfit_'+str(num_bases)+'_'+str(num_params) + '.zip')

    pca_model_loc = os.path.join(model_dir, 'pcaModel.pickle')

    emulator = GPEmulator(gp_model_loc=gp_model_loc, pca_model_loc=pca_model_loc)
    y = emulator(params)
    return y


def test_sklearn(params):
    model_dir = 'notebooks'
    model_loc = os.path.join(model_dir, 'mass_emulator.pickle')
    model = SKLearnEmulator(model_loc)
    y = model(params)
    return y
    
    
if __name__ == '__main__':

    save_loc = "/media/mike/internal/agnfinder"
    with h5py.File(os.path.join(save_loc, 'fsps_cache.hdf5'), 'r') as f:
        X = f['fsps_cache']['X'][:10]
        Y = f['fsps_cache']['Y'][:10]

    params = X[0]
    
    spectra = test_gp(params)
    plt.loglog(spectra)
    plt.savefig('emulated_spectra.png')
    
    mass = test_sklearn(params)
    print(mass)
