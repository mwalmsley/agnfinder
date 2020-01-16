import numpy as np

def gaussian_loglikelihood(model, mag, *params):
    try:
        expected = model(params)
        return -np.abs(expected - mag) # i.e. Gaussian noise with sigma=1
    except ValueError:  # outside of model bounds
        return -np.inf

