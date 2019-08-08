import os
import logging
import argparse

import numpy as np
import tensorflow as tf
import h5py
import corner
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.priors as priors
from agnfinder.prospector.main import save_samples

tf.logging.set_verbosity(tf.logging.ERROR)

def demo_simulator(theta, seed=1, simulator_args=None, batch=1):
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
    sim_data = np.vstack([demo_simulator(sim_params[n, :]) for n in range(len(sim_params))])

    # define prior
    lower = np.zeros(n_parameters)
    upper = np.ones(n_parameters)
    # dimension of prior should be ...?
    prior = priors.Uniform(lower, upper)

    return observed_data, sim_params, sim_data, lower , upper, prior, None


def real():

    data_loc = 'data/photometry_simulation_100000.hdf5'
    logging.warning('Using data loc {}'.format(data_loc))
    assert os.path.isfile(data_loc)
    with h5py.File(data_loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        simulated_y = f['samples']['simulated_y'][...]

    # test set observation
    test_index = int(len(theta) / 3)
    logging.info('Testing simulated galaxy {}'.format(test_index))
    true_params = theta[test_index]
    true_observation = simulated_y[test_index]
    
    # training set
    sim_params = np.vstack([theta[:test_index], theta[test_index+1:]])
    sim_data = np.vstack([simulated_y[:test_index], simulated_y[test_index+1:]])

    # prior is uniform in normalised hypercube space
    # will be log-uniform in real space for log params, but that's fine
    lower = np.zeros(sim_params.shape[1])
    upper = np.ones(sim_params.shape[1])
    prior = priors.Uniform(lower, upper)    

    return true_observation, sim_params, sim_data, lower , upper, prior, true_params


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train pydelfi')
    parser.add_argument('data_loc', type=str)
    parser.add_argument('--test', default=False, dest='test', action='store_true')
    parser.add_argument('--emulate-ssp', default=False, dest='emulate_ssp', action='store_true')
    args = parser.parse_args()

    model_size = 'smaller'  # TODO properly explore possibilities
    name = 'presim_{}_test_{}'.format(model_size, args.test)
    output_dir = os.path.join('results/lfi', name)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(
        filename=os.path.join(output_dir, '{}.log'.format(name)),
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    # construct problem for delfi. Encapsultes physics. This should be a class.
    if args.test:
        observed_data, sim_params, sim_data, lower, upper, prior, true_params = demo()
    else:
        observed_data, sim_params, sim_data, lower, upper, prior, true_params = real()

    epochs = 100

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
    DelfiEnsemble.train_ndes(epochs=epochs)

    posterior_samples = DelfiEnsemble.emcee_sample()
    save_samples(posterior_samples, os.path.join(output_dir, 'samples.hdf5'))
    filtered_samples = posterior_samples[(posterior_samples > 0.) & (posterior_samples < 1.)]
    figure = corner.corner(posterior_samples)
    # TODO corner plot labels
    # , labels=['a', 'b'],
                        # show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(output_dir, 'corner.png'))

    if true_params is not None:
        # TODO automated comparison
        print(true_params)
        logging.info(true_params)
