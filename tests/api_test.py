import pytest

import tensorflow as tf
import numpy as np

from agnfinder.tf_sampling import api

def test_is_out_of_bounds():
    x = np.random.rand(4, 6)  # i.e. 4 chains, 6 params
    x[1, 0] = 1.1  # chain 1, param 0 is OOB, all rest are fine
    x = tf.constant(x)
    
    result = api.is_out_of_bounds(x)
    with tf.Session() as sess:
        result = sess.run(result)
    assert (np.array(result) == np.array([False, True, False, False])).all()


def test_get_log_prob_fn():
    n_chains = 4
    observation_dim = 5
    param_dim = 3
    true_observation = np.random.rand(observation_dim)
    forward_model = lambda x, training: np.random.rand(n_chains, observation_dim)

    log_prob_fn = api.get_log_prob_fn(
        forward_model=forward_model,
        true_observation=true_observation,
        batch_dim=n_chains
    )

    good_params = np.random.rand(n_chains, param_dim)
    good_log_prob = log_prob_fn(good_params)

    bad_params = good_params 
    bad_params[1, 0] = 1.1 # zeroth param of chain 1 is out-of-bounds
    bad_log_prob = log_prob_fn(bad_params)
    
    with tf.Session() as sess:
        bad_log_prob, good_log_prob = sess.run([bad_log_prob, good_log_prob])
    print(bad_log_prob, good_log_prob)
    assert bad_log_prob[1] < good_log_prob[1] - 500  # to avoid random passing, must be MUCH lower as per the large penalty