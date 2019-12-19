import os
import logging
import argparse

import numpy as np
import tensorflow as tf

from agnfinder.tf_sampling import run_sampler, deep_emulator

if __name__ == '__main__':
    """
    Expects model to already be trained and placed in `checkpoint_loc`

    Example use:
        python agnfinder/tf_sampling/run_sampler_parallel.py  --index $INDEX --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling
    """

    parser = argparse.ArgumentParser(description='Run emulated HMC on one galaxy in one thread')
    parser.add_argument('--index', type=int, dest='index')
    parser.add_argument('--cube-loc', type=str, dest='cube_loc', default='Use the cube')  # TODO
    parser.add_argument('--catalog-loc', type=str, dest='catalog_loc', default=None)  # maybe is None anyway
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc')
    parser.add_argument('--output-dir', dest='output_dir', type=str)  # in which save_dir while be created
    parser.add_argument('--n-burnin', type=int, default=1500, dest='n_burnin')  # below 1000, may not find good step size
    parser.add_argument('--n-samples', type=int, default=6000, dest='n_samples')  # 6000 works well?
    parser.add_argument('--n-chains', type=int, default=96, dest='n_chains')  # 96 is ideal on my laptop, more memory = more chains free
    parser.add_argument('--init', type=str, dest='init_method', default='optimised', help='Can be one of: random, roughly_correct, optimised')
    args = parser.parse_args()

    tf.enable_eager_execution()  # for now, this is required

    if args.cube_loc:  # use hypercube
        assert args.catalog_loc is None
        _, _, x_test, y_test = deep_emulator.data()  # catalog_loc will be passed into here
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        true_params = x_test[args.index]
        true_observation = y_test[args.index]
        # to preserve memory
        del x_test
        del y_test

    else:  # use data specified in photometry cube
        assert args.cube_loc is None
        true_params = None
        true_observation = 'TODO' # load from appropriate hdf5 index

    run_sampler.run_on_single_galaxy(
        name=args.index,
        true_observation=true_observation,
        true_params=true_params,
        emulator=deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), args.checkpoint_loc, new=False),
        n_burnin=args.n_burnin,
        n_samples=args.n_samples,
        n_chains=args.n_chains,
        init_method=args.init_method,
        save_dir=args.output_dir
    )