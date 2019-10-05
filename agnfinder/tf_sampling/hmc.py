import os
import logging
import json
import datetime
import argparse

import corner
import numpy as np
from pydelfi import ndes
import tensorflow as tf
import tensorflow_probability as tfp

from agnfinder.tf_sampling import deep_emulator
from agnfinder.tf_sampling.api import Sampler, SamplingProblem, get_log_prob_fn


class SamplerHMC(Sampler):

    def __init__(self, problem: SamplingProblem, n_burnin, n_samples, n_chains, init_method='random'):
        assert tf.executing_eagerly()  # required for sampling
        self.problem = problem
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        self.n_chains = n_chains
        assert init_method in {'random', 'optimised', 'correct', 'roughly_correct'}
        self.init_method = init_method

    def sample(self):
        log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation, batch_dim=self.n_chains)

        if self.init_method == 'correct':
            true_params_stacked = np.vstack([self.problem.true_params.astype(np.float32) for n in range(self.n_chains)])
            initial_state = tf.constant(true_params_stacked)
        elif self.init_method == 'roughly_correct':
            not_quite_true_params = np.vstack([self.problem.true_params for n in range(self.n_chains)] + np.random.rand(self.n_chains, len(self.problem.true_params)) * 0.03).astype(np.float32)
            initial_state = tf.constant(not_quite_true_params)
        elif self.init_method == 'random':
            initial_state = many_random_starts(self.problem.forward_model, self.problem.true_observation, self.problem.true_params, self.n_chains)
        elif self.init_method == 'optimised':
            initial_state = optimized_start(self.problem.forward_model, self.problem.true_observation, self.problem.n_dim, self.n_chains, steps=3000)
        else:
            raise ValueError('Initialisation method {} not recognised'.format(self.init_method))

        print('Ready to go - beginning sampling at {}'.format(datetime.datetime.now().ctime()))
        start_time = datetime.datetime.now()
        samples, is_accepted = hmc(
            log_prob_fn=log_prob_fn,
            initial_state=initial_state,
            num_results=self.n_samples,
            num_burnin_steps=self.n_burnin
        )

        flat_samples = samples.numpy().reshape(-1, 7)
        within_bounds = (np.max(flat_samples, axis=1) < 1.) & (np.min(flat_samples, axis=1) > 0.)
        flat_samples = flat_samples[within_bounds]
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        ms_per_sample = 1000 * elapsed.total_seconds() / (self.n_samples * self.n_chains)  # not counting burn-in as a sample, so really quicker
        print('Sampling {} x {} chains complete in {}, {} ms per sample'.format(self.n_samples, self.n_chains, elapsed, ms_per_sample))
        
        print('Acceptance ratio: {:.4f}'.format(is_accepted.numpy()))
        if is_accepted < 0.01:
            logging.critical('HMC failed to adapt - is step size too small? Is there some burn-in?')
        elif is_accepted < 0.3:
            print('Warning - acceptance ratio is low!')

        print('True vs. median recovered parameters: ', list(zip(self.problem.true_params, np.median(flat_samples, axis=0))))
        return flat_samples


def hmc(log_prob_fn, initial_state, num_results=int(10e3), num_burnin_steps=int(1e3)):

    # Initialize the HMC transition kernel.
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            num_leapfrog_steps=3,
            step_size=2,
            state_gradients_are_stopped=True),
        num_adaptation_steps=int(num_burnin_steps * 0.8))
    assert tf.executing_eagerly()

    @tf.function
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

        is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
        return samples, is_accepted

    samples, is_accepted = run_chain()

    return samples, is_accepted


def find_best_params(x, forward_model, observation, steps, optimizer=tf.train.AdamOptimizer(learning_rate=1e-2)):
    log_prob_fn = get_log_prob_fn(forward_model, observation, batch_dim=tf.shape(x)[0])
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss_value = -log_prob_fn(x)  # a very important minus sign...
            grads = tape.gradient(loss_value, [x])[0]
            grads_and_vars = [(grads, x)]
            optimizer.apply_gradients(grads_and_vars)
    return x


def optimized_start(forward_model, observation, param_dim, n_chains, steps=1000):
    params = tf.Variable(np.ones((1, param_dim), dtype=np.float32) * 0.5, dtype=tf.float32)
    best_params = find_best_params(forward_model, params, observation, steps)
    initial_state = tf.reshape(tf.stack([best_params for n in range(n_chains)]), (n_chains, param_dim))
    return initial_state


def many_random_starts(forward_model, true_observation, true_params, n_chains):
    overproposal_factor = 10
    overproposed_initial_state = tf.random.uniform(shape=(n_chains * overproposal_factor, len(true_params)))
    log_prob_fn = get_log_prob_fn(forward_model, true_observation, batch_dim=n_chains * overproposal_factor)
    start_time = datetime.datetime.now()
    initial_log_probs = log_prob_fn(overproposed_initial_state)
    end_time = datetime.datetime.now()
    ms_elapsed = (end_time - start_time).total_seconds() * 1000
    ms_per_sample =  ms_elapsed / (n_chains * overproposal_factor)
    print('{} samples at {} ms per sample'.format(n_chains*overproposal_factor, ms_per_sample))
    initial_state = tf.gather(overproposed_initial_state, tf.argsort(initial_log_probs))[int((overproposal_factor - 1) * n_chains):]
    return initial_state
