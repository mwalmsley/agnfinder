import os

import numpy as np
import tensorflow as tf
import h5py
import corner
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.priors as priors
from agnfinder.prospector.main import save_samples

tf.logging.set_verbosity(tf.logging.ERROR)

def simulator(theta, seed=1, simulator_args=None, batch=1):
    return np.array([np.mean(theta) + 0.1 * np.random.rand()])

compressor_args=None
def compressor(d, compressor_args):
    # pydelfi can compress observables, but you might not need this for input dimension < 10ish
    return d


if __name__ == '__main__':

    output_dir = 'data/lfi/smaller'
    simulator_args = None
    
    n_parameters = 2  # number of parameters
    n_observables = 1  # number of observables

    # specify the experimental observations
    observed_data = [.3]

    # define prior
    lower = np.zeros(n_parameters)
    upper = np.ones(n_parameters)
    # dimension of prior should be ...?
    prior = priors.Uniform(lower, upper)

    # big model
    # n_hiddens = [50, 50, 50, 50]
    # n_maf = 10

    # small model
    n_hiddens = [20, 20]
    n_maf = 1

    # Build list of neural networks
    NDEs = [
        ndes.ConditionalMaskedAutoregressiveFlow(
            n_parameters=n_parameters,
            n_data=n_observables,
            n_hiddens=n_hiddens,
            n_mades=2,
            act_fun=tf.tanh,
            index=index
        )
        for index in range(n_maf)
    ]

    DelfiEnsemble = delfi.Delfi(observed_data, prior, NDEs,
                                    param_limits=[lower, upper],
                                    param_names=['M1', 'TB', 'Mu', 'M2', 'Msl'],
                                    results_dir= output_dir + "/",
                                    progress_bar=True,
                                   )

    # total number of points to evaluate
    total_sims = 500

    # points to evaluate with uniform priors (before training)
    n_initial = 125 

    # number of additional training runs
    n_populations = 3

    # size of each batch
    n_batch = int((total_sims-n_initial)/n_populations) 
    # with this, we have 3 runs with 100 points in each

    # sometimes (many times) chosen parameters are unphysical or too far
    # outside of our area of interest. In this case, the model returns [-inf, -inf]
    # Therefore, the model must have some backup points to evaluate at instead
    # safety is the number of backup points
    safety = int(9999/n_batch)  # must be less than 10k

    # number of times each point in the batch is used to train the neural networks
    n_epochs = 15 # 15 is small, usually a few hundred is good

    DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations,
                                  patience=max(1, int(3*n_epochs/20)),  # patience during each NN training sesh
                                  epochs=n_epochs,
                                  safety=safety,
                                  save_intermediate_posteriors=False)

    posterior_samples = DelfiEnsemble.emcee_sample()
    save_samples(posterior_samples, os.path.join(output_dir, 'samples.hdf5'))
    filtered_samples = posterior_samples[(posterior_samples > 0.) & (posterior_samples < 1.)]
    figure = corner.corner(posterior_samples)
    # , labels=['a', 'b'],
                        # show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(output_dir, 'corner.png'))
