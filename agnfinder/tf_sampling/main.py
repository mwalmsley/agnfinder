import os
import json
import datetime
import argparse

import corner
import numpy as np
from pydelfi import ndes
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from agnfinder.tf_sampling import deep_emulator, api
from agnfinder.tf_sampling.api import SamplingProblem
from agnfinder.tf_sampling.hmc import SamplerHMC


def test_log_prob_fn(problem):
    # test this TODO

    true_params_2d = tf.reshape(problem.true_params, (1, 7))

    # plt.figure()
    # plt.scatter(range(12), problem.forward_model(true_params_2d), label='model prediction')
    # plt.scatter(range(12), problem.true_observation, label='full simulation (truth)')
    # plt.xlabel('Band (ordered indices)')
    # plt.ylabel('Predicted mag')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('results/model_vs_sim_at_logp_test_case.png')

    log_prob_fn = api.get_log_prob_fn(problem.forward_model, problem.true_observation, batch_dim=1)
    plt.figure()

    print(log_prob_fn(true_params_2d))
    print(log_prob_fn(0.001 + true_params_2d))
    print(log_prob_fn(-0.001 + true_params_2d))
    print(log_prob_fn(0.01 + true_params_2d))
    print(log_prob_fn(-0.01 + true_params_2d))
    print(log_prob_fn(0.1 + true_params_2d))
    print(log_prob_fn(-0.1 + true_params_2d))


    param_index = 0
    param_values = np.linspace(0.01, 0.99, 100)
    log_prob = np.zeros(100)
    for n in range(100):
        new_param = param_values[n].reshape(1)
        # print(new_param, 'new')
        other_params = true_params_2d[0, 1:].numpy()
        # print(other_params, 'other')
        modified_value = np.concatenate((new_param, other_params))
        modified_params = tf.Variable(modified_value.reshape(1, 7))
        # print(modified_params)
        # modified_params[0, param_index] = param_values[n]
        result = log_prob_fn(modified_params).numpy()
        # print(result)
        log_prob[n] = result
    plt.plot(param_values, log_prob, label='Log prob')
    plt.xlabel('Param {}'.format(param_index))
    plt.ylabel('Log prob with modified param')
    print(true_params_2d[0, param_index])
    # plt.axvline(true_params_2d[0, param_index], label='True')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/log_prob_modified_param.png')


    
    exit()

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

    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'  # must match saved checkpoint of emulator
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=new_emulator)

    with open('data/lfi_test_case.json', 'r') as f:
        test_pair = json.load(f)
        true_params = np.array(test_pair['true_params']).astype(np.float32)
        true_observation = np.array(test_pair['true_observation']).astype(np.float32)

    problem = SamplingProblem(true_observation, true_params, forward_model=emulator)

    # test this TODO
    test_log_prob_fn(problem)  # seems like the real parameters are very unlikely and not a minima for the log prob fn (at least nearby). sad times!
    
    sampler = SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method='roughly_correct')
    flat_samples = sampler()

    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    figure = corner.corner(flat_samples, labels=labels)  # middle dim is per chain
    figure.savefig('results/samples_{}_then_{}x{}.png'.format(n_burnin, n_samples, n_chains))

    exit()  # avoids weird tf.function error
