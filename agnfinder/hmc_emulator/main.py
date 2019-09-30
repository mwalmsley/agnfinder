import os
import json
import datetime
import argparse

import corner
import numpy as np
from pydelfi import ndes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_probability as tfp

from agnfinder.hmc_emulator import deep_emulator


def hmc(log_prob_fn, initial_state, num_results=int(10e3), num_burnin_steps=int(1e3)):

    # Initialize the HMC transition kernel.
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            num_leapfrog_steps=3,
            step_size=2.,
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


def find_best_params(x, observation, steps, optimizer=tf.train.AdamOptimizer(learning_rate=1e-2)):
    log_prob_fn = get_log_prob_fn(observation, batch_dim=tf.shape(x)[0])
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss_value = -log_prob_fn(x)  # a very important minus sign...
            grads = tape.gradient(loss_value, [x])[0]
            grads_and_vars = [(grads, x)]
            optimizer.apply_gradients(grads_and_vars)
    return x


def optimized_start(observation, param_dim, n_chains, steps=1000):
    params = tf.Variable(np.ones((1, param_dim), dtype=np.float32) * 0.5, dtype=tf.float32)
    best_params = find_best_params(params, observation, steps)
    initial_state = tf.reshape(tf.stack([best_params for n in range(n_chains)]), (n_chains, param_dim))
    return initial_state


def many_random_starts():
    overproposal_factor = 1000
    overproposed_initial_state = tf.random.uniform(shape=(n_chains * overproposal_factor, len(true_params)))
    log_prob_fn = get_log_prob_fn(true_observation, batch_dim=n_chains * overproposal_factor)
    start_time = datetime.datetime.now()
    initial_log_probs = log_prob_fn(overproposed_initial_state)
    end_time = datetime.datetime.now()
    ms_elapsed = (end_time - start_time).total_seconds() * 1000
    ms_per_sample =  ms_elapsed / (n_chains * overproposal_factor)
    print('{} samples at {} ms per sample'.format(n_chains*overproposal_factor, ms_per_sample))
    initial_state = tf.gather(overproposed_initial_state, tf.argsort(initial_log_probs))[int((overproposal_factor - 1) * n_chains):]
    return initial_state


def get_log_prob_fn(true_observation, batch_dim):
    true_observation_stacked = tf.stack([tf.constant(true_observation) for n in range(batch_dim)])
    # first dimension of true params must match first dimension of x, or will fail
    def log_prob_fn(x):
        expected_photometry = -emulator(x, training=False)  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(10 ** expected_photometry - 10 ** true_observation_stacked)
        sigma = (10 ** expected_photometry) * 0.05
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)
        return log_prob
    return log_prob_fn


def run_hmc_sampling(true_observation, true_params, emulator, n_chains, n_samples, n_burnin):
    log_prob_fn = get_log_prob_fn(true_observation, batch_dim=n_chains)

    # exactly correct start
    true_params_stacked = np.vstack([true_params.astype(np.float32) for n in range(n_chains)])
    initial_state = tf.constant(true_params_stacked)

    # roughly correct start
    # not_quite_true_params = np.vstack([true_params for n in range(n_chains)] + np.random.rand(n_chains, len(true_params)) * 0.03).astype(np.float32)
    # initial_state = tf.constant(not_quite_true_params)

    # random start
    # initial_state = many_random_starts()

    # optimized start
    # print(true_params)
    # initial_state = optimized_start(len(true_params), n_chains, steps=5000)


    # print('Params')
    # print(true_params)
    # print(initial_state)

    print('Observations')
    print(true_observation)
    print(-emulator(true_params.reshape(1, -1), training=False))


    print('Ready to go - beginning sampling at {}'.format(datetime.datetime.now().ctime()))
    start_time = datetime.datetime.now()
    samples, is_accepted = hmc(
        log_prob_fn=log_prob_fn,
        initial_state=initial_state,
        num_results=n_samples,
        num_burnin_steps=n_burnin
    )
    print(samples.shape)
    within_bounds = (np.max(samples, axis=2) < 1.) & (np.min(samples, axis=2) > 0.)
    print(within_bounds.shape)
    samples = samples[within_bounds]
    print(len(samples))
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    ms_per_sample = 1000 * elapsed.total_seconds() / (n_samples * n_chains)  # not counting burn-in as a sample, so really quicker
    print('Sampling {} x {} chains complete in {}, {} ms per sample'.format(n_samples, n_chains, elapsed, ms_per_sample))

    flat_samples = samples.numpy().reshape(-1, 7)
    print(is_accepted)
    print(list(zip(true_params, np.median(flat_samples, axis=0))))
    return flat_samples

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Sample emulator')
    parser.add_argument('--new-emulator', default=False, dest='new_emulator', action='store_true')
    parser.add_argument('--n-chains', type=int, default=16, dest='n_chains')
    parser.add_argument('--n-samples', type=int, default=int(1e3), dest='n_samples')
    parser.add_argument('--n-burnin', type=int, default=300, dest='n_burnin')
    args = parser.parse_args()

    new_emulator = args.new_emulator
    n_chains = args.n_chains
    n_samples = args.n_samples
    n_burnin = args.n_burnin

    checkpoint_loc = 'results/checkpoints/trained_deep_emulator'  # must match saved checkpoint of emulator
    emulator = deep_emulator.get_trained_emulator(deep_emulator.tf_model(), checkpoint_loc, new=new_emulator)

    with open('data/lfi_test_case.json', 'r') as f:
        test_pair = json.load(f)
        true_params = np.array(test_pair['true_params']).astype(np.float32)
        true_observation = np.array(test_pair['true_observation']).astype(np.float32)

    
    flat_samples = run_hmc_sampling(true_observation, true_params, emulator, n_chains, n_samples, n_burnin)

    # TODO flat_samples = run_nested_sampling(true_observation, true_params, emulator, **nested_sampling_args)

    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    figure = corner.corner(flat_samples, labels=labels)  # middle dim is per chain
    figure.savefig('results/samples_{}_then_{}x{}.png'.format(n_burnin, n_samples, n_chains))

    exit()  # avoids weird tf.function error