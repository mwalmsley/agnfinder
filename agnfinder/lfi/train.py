import os
import logging
import argparse
import json
import datetime
import pickle

import numpy as np
import tensorflow as tf
import h5py
import corner
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
from agnfinder.prospector.main import save_samples

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    sim_data = np.vstack([demo_simulator(sim_params[n, :])
                          for n in range(len(sim_params))])

    # define prior
    lower = np.zeros(n_parameters)
    upper = np.ones(n_parameters)
    # dimension of prior should be ...?
    prior = priors.Uniform(lower, upper)

    return observed_data, sim_params, sim_data, lower, upper, prior, None


class CustomDelfi(delfi.Delfi):

    def saver(self):
        f = open(self.restore_filename, 'wb')
        pickle.dump([self.stacking_weights, self.posterior_samples, self.proposal_samples, self.training_loss, self.validation_loss, self.stacked_sequential_training_loss, self.stacked_sequential_validation_loss, self.sequential_nsims, self.ps, self.xs, self.x_mean, self.x_std, self.p_mean, self.p_std], f)
        f.close()

        # print(self.nde)
        # print(self.nde[0])
        print(self.nde[0].data)
        print(self.nde[0].u)


        # all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # all_vars = tf.all_variables()
        # all_vars = [n for n in tf.get_default_graph().as_graph_def().node if n.name.startswith('nde_0_made_1')]
        # for v in all_vars:
        #     # if v.name == 'nde_0_made_1/m':
        #     if 'data' in v.name:
        #         print(v)
        # exit()

        

        nde_scope = 'nde_0_made_1'
        outputs = {'m': self.nde[0].mades[0].m}
        inputs = {'data': self.nde[0].data}
        
        export_dir = os.path.join(self.results_dir, 'saved_model')
        tf.compat.v1.saved_model.simple_save(
            self.sess,
            export_dir,
            inputs,
            outputs
        )
    

# TODO python 3.7 dataclass would be great here
class DelfiProblem():

    def __init__(
        self,
        true_observation,
        true_params,
        lower,
        upper,
        prior,
        sim_data=None,
        sim_params=None,
        simulator=None
    ):

        # not yet supported to do both previous and new sims
        assert (sim_params is None) != (simulator is None)
        self.true_observation = true_observation
        self.true_params = true_params
        self.sim_data = sim_data
        self.sim_params = sim_params
        self.simulator = simulator
        self.lower = lower
        self.upper = upper
        self.prior = prior

    @property
    def param_dim(self):
        return len(self.true_params)

    @property
    def observable_dim(self):
        return len(self.true_observation)


def solve(problem, output_dir, epochs=1, model_size='smaller', n_posterior_samples=1):

    if model_size == 'bigger':
        n_hiddens = [50, 50, 50, 50]
        n_maf = 5
    elif model_size == 'smaller':
        n_hiddens = [20, 20]
        n_maf = 3
    else:
        raise ValueError(model_size)

    # Build list of neural networks
    NDEs = [
        ndes.ConditionalMaskedAutoregressiveFlow(
            n_parameters=problem.param_dim,
            n_data=problem.observable_dim,
            n_hiddens=n_hiddens,
            n_mades=2,
            act_fun=tf.tanh,
            index=index
        )
        for index in range(n_maf)
    ]

    delfi_ensemble = CustomDelfi(problem.true_observation, problem.prior, NDEs,
                                param_limits=[problem.lower, problem.upper],
                                param_names=['param_{}'.format(
                                    n) for n in range(problem.param_dim)],
                                results_dir=output_dir + "/",
                                progress_bar=True,
                                # n_procs=10,
                                )

    if problem.sim_data is not None:
        delfi_ensemble.load_simulations(problem.sim_data, problem.sim_params)
        delfi_ensemble.train_ndes(epochs=epochs)
    elif problem.simulator is not None:

        def compressor(data, *args):
            return data  # do nothing. pydelfi should really allow compressor=None - submit a PR? TODO

        # total_sims = 50000  # num. points to evaluate
        # num. points to evaluate with uniform priors (before training)
        n_initial = 10000
        n_populations = 5  # number of additional training runs
        # size of each batch
        # n_batch = int((total_sims-n_initial)/n_populations)
        n_batch = 9999
        n_epochs = 100

        safety = 5
        # Overpropose by a factor of safety to (hopefully) cope gracefully with
        # the possibility of some bad proposals.

        # from mpi4py import MPI
        # comm = MPI.COMM_WORLD
        # rank = comm.Get_rank()
        # n_procs = 10

        # comm = None
        # rank = None
        # n_procs = 1


        # bayesian optimisation is another option here , to investigate TODO
        patience = max(1, int(3*n_epochs/20))
        delfi_ensemble.sequential_training(
            problem.simulator,
            compressor,
            n_initial,
            n_batch,
            n_populations,
            patience=patience,  # patience during each NN training sesh
            epochs=n_epochs,
            safety=safety,
            save_intermediate_posteriors=False)

    n_posterior_samples = 1
    print('All training complete. Sampling posterior ({} samples)'.format(n_posterior_samples))
    start_time = datetime.datetime.now()
    posterior_samples = delfi_ensemble.emcee_sample(main_chain=n_posterior_samples)
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    logging.info(end_time - start_time)
    save_samples(posterior_samples, os.path.join(output_dir, 'samples.hdf5'))
    # filtered_samples = posterior_samples[(
    #     posterior_samples > 0.) & (posterior_samples < 1.)]
    figure = corner.corner(posterior_samples)
    # TODO corner plot labels
    # , labels=['a', 'b'],
    # show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig(os.path.join(output_dir, 'corner.png'))

    if problem.true_params is not None:
        # TODO automated comparison
        print(problem.true_params)
        logging.info(problem.true_params)

