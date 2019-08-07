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

def demo():
    n_parameters = 2  # number of parameters
    n_observables = 1  # number of observables

    # specify the experimental observations
    observed_data = [.3]

    # specify previous simulations
    n_simulated_points = 10000
    sim_params = np.random.rand(n_simulated_points, n_parameters)
    assert n_observables == 1
    sim_data = np.vstack([simulator(sim_params[n, :]) for n in range(len(sim_params))])

    # define prior
    lower = np.zeros(n_parameters)
    upper = np.ones(n_parameters)
    # dimension of prior should be ...?
    prior = priors.Uniform(lower, upper)

    return observed_data, sim_params, sim_data, lower , upper, prior, None


def real():

    data_loc = '/media/mike/internal/agnfinder/photometry_simulation.hdf5'
    assert os.path.isfile(data_loc)
    with h5py.File(data_loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        simulated_y = f['samples']['simulated_y'][...]

    # test set observation
    test_index = 5
    true_params = theta[test_index]
    true_observation = simulated_y[test_index]
    
    # training set
    sim_params = np.vstack([theta[:test_index], theta[test_index+1:]])
    sim_data = np.vstack([simulated_y[:test_index], simulated_y[test_index+1:]])

    # prior is uniform in normalised hypercube space
    # will be log-uniform in real space for log params, but that's fine
    lower = np.zeros(sim_params.shape[1])
    upper = np.ones(sim_params.shape[1])
    # dimension of prior should be ...?
    prior = priors.Uniform(lower, upper)    

    print(true_observation.shape, sim_data.shape, sim_params.shape, lower.shape, upper.shape)

    return true_observation, sim_params, sim_data, lower , upper, prior, true_params


compressor_args=None
def compressor(d, compressor_args):
    # pydelfi can compress observables, but you might not need this for input dimension < 10ish
    return d


if __name__ == '__main__':

    model_size = 'smaller'
    test = False
    output_dir = 'results/lfi/presim_{}_test_{}'.format(model_size, test)
    simulator_args = None

    if test:
        observed_data, sim_params, sim_data, lower, upper, prior, true_params = demo()
    else:
        observed_data, sim_params, sim_data, lower, upper, prior, true_params = real()


    n_parameters = sim_params.shape[1]
    n_observables = sim_data.shape[1]
    
    if model_size == 'bigger':
        n_hiddens = [50, 50, 50, 50]
        n_maf = 10
    elif model_size == 'smaller':
        n_hiddens = [20, 20]
        n_maf = 3
    else:
        raise ValueError(model_size)

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
                                    param_names=['param_{}'.format(n) for n in range(n_parameters)],
                                    results_dir= output_dir + "/",
                                    progress_bar=True,
                                   )
    DelfiEnsemble.load_simulations(sim_data, sim_params)
    DelfiEnsemble.train_ndes()

    posterior_samples = DelfiEnsemble.emcee_sample()
    save_samples(posterior_samples, os.path.join(output_dir, 'samples.hdf5'))
    filtered_samples = posterior_samples[(posterior_samples > 0.) & (posterior_samples < 1.)]
    figure = corner.corner(posterior_samples)
    # , labels=['a', 'b'],
                        # show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(output_dir, 'corner.png'))

    if true_params is not None:
        print(true_params)
