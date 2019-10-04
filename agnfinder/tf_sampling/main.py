import os
import json
import datetime
import argparse

import corner
import numpy as np
from pydelfi import ndes
import tensorflow as tf
import tensorflow_probability as tfp

from agnfinder.tf_sampling import deep_emulator
from agnfinder.tf_sampling.api import SamplingProblem
from agnfinder.tf_sampling.hmc import SamplerHMC

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
    
    sampler = SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method='roughly_correct')
    flat_samples = sampler()

    labels = ['mass', 'dust2', 'tage', 'tau', 'agn_disk_scaling', 'agn_eb_v', 'agn_torus_scaling']
    figure = corner.corner(flat_samples, labels=labels)  # middle dim is per chain
    figure.savefig('results/samples_{}_then_{}x{}.png'.format(n_burnin, n_samples, n_chains))

    exit()  # avoids weird tf.function error
