import pytest

import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from agnfinder.tf_sampling import hmc, deep_emulator


def test_find_minima():

    tf.compat.v1.enable_eager_execution()

    observation = tf.constant([0.3, 0.4], dtype=tf.float32)
    func = lambda x: tf.reduce_sum(input_tensor=tf.abs(x * 2. - observation))
    initial_guess_np = [0.7, 0.1]
    best_params = tf.Variable(initial_guess_np)
    steps = 1000  # needs to be about this many, adam takes a lot of steps with default learning rate TODO

    hmc.find_minima(func, best_params, steps)  # acts in-place
    expected_params = observation.numpy() / 2.

    print('Initial: {}'.format(initial_guess_np))
    print('Final: {}'.format(best_params.numpy()))
    print('True: {}'.format(expected_params))

    assert (np.abs(best_params.numpy() - expected_params) < 0.1).all()

    # plt.plot(*initial_guess_np, 'x', label='Initial')
    # plt.plot(*best_params.numpy(), 'x', label='Final')
    # plt.plot(*expected_params, 'x', label='True')
    # plt.legend()
    # plt.show()


def test_find_best_params(monkeypatch):

    tf.compat.v1.enable_eager_execution()

    n_chains = 1
    param_dim = 2
    observation = tf.constant([0.3, 0.4], dtype=tf.float32)
    forward_model = lambda x, training: x * 2.

    def mock_get_log_prob_fn(x, y, batch_dim):
        return lambda x: log_prob_fn(x)

    log_prob_fn = lambda x: -tf.reduce_sum(input_tensor=tf.abs(forward_model(x, training=None) - observation))
    monkeypatch.setattr(hmc, 'get_log_prob_fn', mock_get_log_prob_fn)
    steps = 1000

    expected_params = observation / 2.
    best_params = hmc.find_best_params(forward_model, observation, param_dim, n_chains, steps)
    print('Final: {}, expected: {}'.format(best_params.numpy(), expected_params))
    assert (np.abs(best_params.numpy() - expected_params) < 0.1).all()

def test_find_best_params_batch(monkeypatch):

    tf.compat.v1.enable_eager_execution()

    n_chains = 4
    param_dim = 2
    observation = tf.constant([0.3, 0.4], dtype=tf.float32)
    forward_model = lambda x, training: x * 2.

    def mock_get_log_prob_fn(x, y, batch_dim):
        return lambda x: log_prob_fn(x)

    log_prob_fn = lambda x: -tf.reduce_sum(input_tensor=tf.abs(forward_model(x, training=None) - observation))
    monkeypatch.setattr(hmc, 'get_log_prob_fn', mock_get_log_prob_fn)
    steps = 1000

    expected_params = observation / 2.
    best_params = hmc.find_best_params(forward_model, observation, param_dim, n_chains, steps)
    print('Final: {}, expected: {}'.format(best_params.numpy(), expected_params))
    assert (np.abs(best_params.numpy() - expected_params) < 0.1).all()


def test_find_best_params_functional():

    tf.compat.v1.enable_eager_execution()

    with open('data/lfi_test_case.json', 'r') as f:
        test_case = json.load(f)
    expected_params = np.array(test_case['true_params'], dtype=np.float32)
    true_observation = -1 * np.array(test_case['true_observation'], dtype=np.float32)

    n_chains = 1
    n_steps = 1000

    # initial_guess = expected_params * 0.8
    initial_guess = np.random.rand(len(expected_params)).astype(np.float32)

    checkpoint_loc = 'results/checkpoints/latest'
    forward_model = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    best_params = hmc.find_best_params(forward_model, true_observation, len(expected_params), n_chains, n_steps, initial_guess=initial_guess)
    print('Final: {}, expected: {}'.format(best_params.numpy(), expected_params))

    fwd_expected = forward_model(expected_params.reshape(1, -1)).numpy().reshape(-1)
    fwd_init = forward_model(initial_guess.reshape(1, -1)).numpy().reshape(-1)
    fwd_best = forward_model(best_params.numpy().reshape(1, -1)).numpy().reshape(-1)

    fig, (ax0, ax1) = plt.subplots(ncols=2)

    x = np.arange(len(expected_params))
    ax0.scatter(x, initial_guess, label='initial')
    ax0.scatter(x, best_params.numpy(), label='best')
    ax0.scatter(x, expected_params, label='true')
    ax0.legend()

    x = np.arange(len(fwd_expected))
    ax1.scatter(x, fwd_init, label='Fwd(initial)')
    ax1.scatter(x, fwd_best, label='Fwd(best)')
    ax1.scatter(x, fwd_expected, label='Fwd(true)')
    ax1.legend()

    fig.tight_layout()
    fig.savefig('tests/test_figures/optimised_start.png')
    # plt.show()

    assert (np.abs(best_params.numpy() - expected_params) < 0.1).all()
