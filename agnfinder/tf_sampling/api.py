import logging

import numpy as np
import tensorflow as tf

from agnfinder.tf_sampling import deep_emulator

class SamplingProblem():

    def __init__(self, true_observation, true_params, forward_model):
        assert true_observation.ndim == 2
        assert true_params.ndim == 2
        self.true_observation = true_observation
        self.true_params = true_params
        self.forward_model = lambda *args, **kwargs: forward_model(*args, **kwargs)

    @property
    def param_dim(self):
        return self.true_params.shape[1]  # 0th is batch, 1st is params


class Sampler():

    def __init__(self, problem: SamplingProblem, *args, **kw_args):
        self.problem = problem

    def sample(self):
        raise NotImplementedError

    def __call__(self):
        return self.sample()


# @tf.function  # inputs become tensors when you wrap
def get_log_prob_fn(forward_model, true_observation):
    assert tf.rank(true_observation).numpy() == 2  # must have batch dim
    # first dimension of true params must match first dimension of x, or will fail
    @tf.function(experimental_compile=True)
    def log_prob_fn(x):  # 0th axis is batch/chain dim, 1st is param dim
        # expected photometry has been normalised by deep_emulator.normalise_photometry, remember - it's neg log10 mags
        expected_photometry = deep_emulator.denormalise_photometry(forward_model(x, training=False))  # model expects a batch dimension, which here is the chains
        true_photometry = true_observation  # make sure you denormalise this in the first place, if loading from data()
        deviation = tf.abs(expected_photometry - true_photometry)
        sigma = expected_photometry * 0.05  # i.e. 5% sigma, will read in soon-ish
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)  # very negative = unlikely, near -0 = likely
        x_out_of_bounds = is_out_of_bounds(x)
        penalty = tf.cast(x_out_of_bounds, tf.float32) * tf.constant(1000., dtype=tf.float32)
        log_prob_with_penalty = log_prob - penalty  # no effect if x in bounds, else divide (subtract) a big penalty
        return log_prob_with_penalty
    return log_prob_fn


def is_out_of_bounds(x):  # x expected to have (batch, param) shape
    return tf.reduce_any( tf.cast(x > 1., tf.bool) | tf.cast(x < 0., tf.bool),  axis=1)
