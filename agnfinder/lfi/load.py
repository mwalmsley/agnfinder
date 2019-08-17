import os
import json
import datetime
import argparse

import corner
import numpy as np
from pydelfi import ndes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_probability as tfp

from agnfinder import deep_emulator


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


def find_best_params(x, steps, optimizer=tf.train.AdamOptimizer(learning_rate=1e-2)):
    for _ in range(steps):
        with tf.GradientTape() as tape:
            loss_value = -log_prob_fn(x)  # a very important minus sign...
            grads = tape.gradient(loss_value, [x])[0]
            grads_and_vars = [(grads, x)]
            optimizer.apply_gradients(grads_and_vars)
    return x


def optimized_start(param_dim, n_chains, steps=1000):
    # @tf.function()
    params = tf.Variable(np.ones((1, param_dim), dtype=np.float32) * 0.5, dtype=tf.float32)
    # print(log_prob_fn(params))
    best_params = find_best_params(params, steps)
    # print(best_params)
    # print(log_prob_fn(best_params))

    initial_state = tf.reshape(tf.stack([best_params for n in range(n_chains)]), (n_chains, param_dim))
    # print(initial_state)
    # print(initial_state.shape)
    # exit()
    return initial_state


def many_random_starts():
    # random start
    overproposal_factor = 1000
    overproposed_initial_state = tf.random.uniform(shape=(n_chains * overproposal_factor, len(true_params)))
    # print('initial state', overproposed_initial_state)
    true_observation_stacked = tf.stack([tf.constant(true_observation) for n in range(n_chains * overproposal_factor)])
    # @tf.function
    def initial_log_prob_fn(x):
        # first dimension of x MUST be chain dimension, e.g. 1 for 1 chain
        expected_photometry = -emulator(x, training=False)  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(10 ** expected_photometry - 10 ** true_observation_stacked)
        sigma = (10 ** expected_photometry) * 0.05
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)
        return log_prob
    start_time = datetime.datetime.now()
    initial_log_probs = initial_log_prob_fn(overproposed_initial_state)
    end_time = datetime.datetime.now()
    ms_elapsed = (end_time - start_time).total_seconds() * 1000
    ms_per_sample =  ms_elapsed / (n_chains * overproposal_factor)
    print('{} samples at {} ms per sample'.format(n_chains*overproposal_factor, ms_per_sample))
    # exit()
    # tf.ragged.boolean_mask(initial_state)
    # print()
    initial_state = tf.gather(overproposed_initial_state, tf.argsort(initial_log_probs))[int((overproposal_factor - 1) * n_chains):]
    # print(initial_state.shape)
    # print(initial_state)
    # exit()
    return initial_state

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Sample emulator')
    parser.add_argument('--n_chains', type=int, default=1, dest='n_chains')
    parser.add_argument('--n_samples', type=int, default=int(1e5), dest='n_samples')
    parser.add_argument('--n_burnin', type=int, default=3000, dest='n_burnin')
    args = parser.parse_args()

    n_chains = args.n_chains
    n_samples = args.n_samples
    n_burnin = args.n_burnin

    # graph_restore_dir = '/media/mike/internal/agnfinder/results/lfi/aligned_data_presim'
    # assert os.path.isdir(graph_restore_dir)

    with open('data/lfi_test_case.json', 'r') as f:
        test_pair = json.load(f)
        true_params = np.array(test_pair['true_params']).astype(np.float32)
        true_observation = np.array(test_pair['true_observation']).astype(np.float32)

    true_observation_stacked_extra = tf.stack([tf.constant(true_observation) for n in range(n_chains)])
    # @tf.function
    def log_prob_fn(x):
        # first dimension of x MUST be chain dimension, e.g. 1 for 1 chain
        expected_photometry = -emulator(x, training=False)  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(10 ** expected_photometry - 10 ** true_observation_stacked_extra)
        sigma = (10 ** expected_photometry) * 0.05
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)
        return log_prob

    
    # Explicit mode
    checkpoint_dir = 'results/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'trained_deep_emulator')
    emulator = deep_emulator.tf_model()
    checkpointer = tf.train.Checkpoint(emulator=emulator)
    new = False
    if new:
        emulator = deep_emulator.train_manual(emulator, *deep_emulator.data())
        print('Training complete - saving')
        checkpointer.save(file_prefix=checkpoint_prefix)
    else:
        print('Loading previous model')
        # print(emulator(true_params.reshape(1, -1)))
        checkpointer = tf.train.Checkpoint(emulator=emulator)
        save_path = tf.train.latest_checkpoint(checkpoint_dir)
        assert save_path is not None
        status = checkpointer.restore(save_path)
        # print(status)
        # print(emulator(true_params.reshape(1, -1)))

    # exit()
    # emulator = deep_emulator.tf_model()



    # exactly correct start
    true_params_stacked = np.vstack([true_params.astype(np.float32) for n in range(n_chains)])
    initial_state = tf.constant(true_params_stacked)
    
    # roughly correct start
    # not_quite_true_params = np.vstack([true_params for n in range(n_chains)] + np.random.rand(n_chains, len(true_params)) * 0.03).astype(np.float32)
    # initial_state = tf.constant(not_quite_true_params)

    # random start
    # initial_state = many_random_starts()
    # print(initial_state.shape)
    # print(initial_state)
    # exit()

    # optimized start
    # print(true_params)
    # initial_state = optimized_start(len(true_params), n_chains, steps=5000)
    # print(initial_state[0])
    # print(initial_state.shape)
    # exit()

    # print('Params')
    # print(true_params)
    # print(initial_state)

    print('Observations')
    print(true_observation)
    print(-emulator(true_params.reshape(1, -1), training=False))
    # print(-emulator(initial_state, training=False))


    # print(log_prob_fn(true_params.reshape(1, -1)))
    # print(log_prob_fn(initial_state))

    # exit()

    # print(initial_state)
    # print(emulator(initial_state, training=False))

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
    # print(samples)True
    # print([log_prob_fn(samples[0:1])])_{}
    # print([log_prob_fn(samples[1000:1001])])
    # print([log_prob_fn(samples[-1:])])

    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    figure = corner.corner(flat_samples, labels=labels)  # middle dim is per chain
    figure.savefig('results/samples_{}_then_{}x{}.png'.format(n_burnin, n_samples, n_chains))

    # with tf.Session(config=tf.ConfigProto()) as sess:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, graph_restore_dir + '/graph_checkpoint')

    #     # print(NDEs[0].eval((true_params.reshape(1, -1), true_observation.reshape(1, -1)), sess))

    #     print(sess.run([sample_mean, is_accepted], feed_dict={
    #         initial_state: true_params.reshape(1, -1),
    #         NDEs[0].data: true_observation.reshape(1, -1)}))
    exit()  # avoids weird tf.function error