import logging

import numpy as np
import tensorflow as tf

from agnfinder.tf_sampling import deep_emulator

class SamplingProblem():

    def __init__(self, true_observation, true_params, forward_model, redshifts, sigma):
        assert true_observation.ndim == 2
        assert true_params.ndim == 2
        self.true_observation = true_observation
        self.true_params = true_params
        self.forward_model = lambda *args, **kwargs: forward_model(*args, **kwargs)
        self.redshifts = redshifts
        self.sigma = sigma

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
def get_log_prob_fn(forward_model, true_photometry, redshifts, sigma):  
    # sigma can be scalar, or (batch_dim, photometry_dim)
    if isinstance(sigma, float):
        uncertainty = true_photometry * sigma  # i.e. 5% sigma
    else:
        assert tf.rank(sigma).numpy() == 2
        uncertainty = sigma
    assert tf.rank(true_photometry).numpy() == 2  # true photometry must be (batch_dim, photometry_dim) even if batch_dim=1
    # first dimension of true params must match first dimension of x, or will fail
    @tf.function(experimental_compile=True)
    def log_prob_fn(x):  # 0th axis is batch/chain dim, 1st is param dim
        # expected photometry has been normalised by deep_emulator.normalise_photometry, remember - it's neg log10 mags
        x_with_redshifts = tf.concat([redshifts, x], axis=1)
        expected_photometry = deep_emulator.denormalise_photometry(forward_model(x_with_redshifts, training=False))  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(expected_photometry - true_photometry)   # make sure you denormalise true observation in the first place, if loading from data(). Should be in maggies.
        log_prob = -tf.reduce_sum(input_tensor=deviation / uncertainty, axis=1)  # very negative = unlikely, near -0 = likely
        x_out_of_bounds = is_out_of_bounds(x)
        penalty = tf.cast(x_out_of_bounds, tf.float32) * tf.constant(1000., dtype=tf.float32)
        log_prob_with_penalty = log_prob - penalty  # no effect if x in bounds, else divide (subtract) a big penalty
        return log_prob_with_penalty
    return log_prob_fn


def is_out_of_bounds(x):  # x expected to have (batch, param) shape
    return tf.reduce_any( input_tensor=tf.cast(x > 1., tf.bool) | tf.cast(x < 0., tf.bool),  axis=1)
