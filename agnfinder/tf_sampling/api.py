import logging

import numpy as np
import tensorflow as tf

class SamplingProblem():

    def __init__(self, true_observation, true_params, forward_model):
        self.true_observation = true_observation
        self.true_params = true_params
        logging.warning('Temporarily adding minus sign to forward model')
        self.forward_model = lambda *args, **kwargs: -forward_model(*args, **kwargs)

    @property
    def n_dim(self):
        return len(self.true_params)


class Sampler():

    def __init__(self, problem: SamplingProblem, *args, **kw_args):
        self.problem = problem

    def sample(self):
        raise NotImplementedError

    def __call__(self):
        return self.sample()


# @tf.function  # inputs become tensors when you wrap
def get_log_prob_fn(forward_model, true_observation, batch_dim=None):
    if batch_dim is not None:
        true_observation_stacked = tf.stack([true_observation for n in range(batch_dim)], axis=0)
    else:
        true_observation_stacked = tf.reshape(true_observation, (1, -1))
    # first dimension of true params must match first dimension of x, or will fail
    @tf.function
    def log_prob_fn(x):  # 0th axis is batch/chain dim, 1st is param dim
        expected_photometry = forward_model(x, training=False)  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(10 ** expected_photometry - 10 ** true_observation_stacked)
        sigma = (10 ** expected_photometry) * 0.05
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)  # very negative = unlikely, near -0 = likely
        x_out_of_bounds = is_out_of_bounds(x)
        penalty = tf.cast(x_out_of_bounds, tf.float32) * tf.constant(1000., dtype=tf.float32)
        log_prob_with_penalty = log_prob - penalty  # no effect if x in bounds, else divide (subtract) a big penalty
        return log_prob
    return log_prob_fn


def is_out_of_bounds(x):  # x expected to have (batch, param) shape
    return tf.reduce_any( tf.cast(x > 1., tf.bool) | tf.cast(x < 0., tf.bool),  axis=1)
