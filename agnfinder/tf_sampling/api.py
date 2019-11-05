import logging

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


# TODO refactor
def get_log_prob_fn(forward_model, true_observation, batch_dim=None):
    if batch_dim is not None:
        true_observation_stacked = tf.stack([tf.constant(true_observation) for n in range(batch_dim)])
    else:
        true_observation_stacked = tf.reshape(true_observation, (1, -1))
    # first dimension of true params must match first dimension of x, or will fail
    def log_prob_fn(x):
        expected_photometry = forward_model(x, training=False)  # model expects a batch dimension, which here is the chains
        deviation = tf.abs(10 ** expected_photometry - 10 ** true_observation_stacked)
        sigma = (10 ** expected_photometry) * 0.05
        log_prob = -tf.reduce_sum(deviation / sigma, axis=1)
        return log_prob
    return log_prob_fn
