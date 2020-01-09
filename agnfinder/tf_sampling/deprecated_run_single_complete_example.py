import os
import logging
import json
import argparse

import corner
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from agnfinder.tf_sampling import deep_emulator, api
from agnfinder.tf_sampling.api import SamplingProblem
from agnfinder.tf_sampling.hmc import SamplerHMC

if __name__ == '__main__':

    """
    Add --new-emulator flag on first run to train the emulator.
    See README.md in this folder for more.

    Example use:
    python agnfinder/tf_sampling/deprecated_run_single_complete_example.py --checkpoint-loc results/checkpoints/latest --test-json data/lfi_test_case.json --n-chains 32 --n-samples 1000 --n-burnin 1500

    """
    tf.enable_eager_execution()

    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    parser = argparse.ArgumentParser(description='Sample emulator')
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc')
    parser.add_argument('--test-json', type=str, dest='test_json_loc', default='data/lfi_test_case.json')
    parser.add_argument('--new-emulator', default=False, dest='new_emulator', action='store_true')
    parser.add_argument('--n-chains', type=int, default=32, dest='n_chains')
    parser.add_argument('--n-samples', type=int, default=int(2e3), dest='n_samples')
    parser.add_argument('--n-burnin', type=int, default=1000, dest='n_burnin')
    args = parser.parse_args()
    checkpoint_loc = args.checkpoint_loc
    assert checkpoint_loc is not None
    test_json_loc = args.test_json_loc
    assert os.path.isfile(test_json_loc)
    new_emulator = args.new_emulator
    n_chains = args.n_chains
    n_samples = args.n_samples
    n_burnin = args.n_burnin

    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=new_emulator)

    with open(test_json_loc, 'r') as f:
        test_pair = json.load(f)
        true_params = np.array(test_pair['true_params']).astype(np.float32)
        true_observation = np.array(test_pair['true_observation']).astype(np.float32)

    problem = SamplingProblem(true_observation, true_params, forward_model=emulator)

    # test_log_prob_fn(problem)  # seems like the real parameters are very unlikely and not a minima for the log prob fn (at least nearby). sad times!
    # exit()
    # init_method = 'random'
    init_method = 'optimised'
    sampler = SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
    samples = sampler()

    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    figure = corner.corner(samples.reshape(-1, 7), labels=labels)  # middle dim is per chain
    figure.savefig('results/samples_{}_then_{}x{}.png'.format(n_burnin, n_samples, n_chains))

    exit()  # avoids weird tf.function error
