import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from nestorflow.nested import nested_sample, samples
import matplotlib.pyplot as plt
import os

from agnfinder.tf_sampling.api import get_log_prob_fn

# def likelihood(x):
#     dist = tfp.distributions.MultivariateNormalDiag(loc=(np.ones(n_dims)*0.5).astype(np.float32), scale_diag=(np.ones(n_dims)*0.1).astype(np.float32))
#     return dist.log_prob(x)


if __name__ == '__main__':

    os.environ['TF_XLA_FLAGS']='--tf_xla_cpu_global_jit'

    n_dims = 1  # params or observables?
    n_live = 10
    # n_repeats = n_dims*2
    n_repeats = 1
    # n_compress = n_dims*5

    # true_observation = np.array([0.5] * 5).astype(np.float32)
    true_observation = 0.5
    # true_params = 0.1

    def forward_model(x, training):
        return x

    # @tf.function
    # def likelihood(x):
        # return tf.expand_dims(tf.log(tf.abs(forward_model(x, training=False) - true_observation)), axis=0)

    # @tf.function
    def likelihood(x):
        return tf.squeeze(tf.math.log(tf.abs(forward_model(x, training=False) - true_observation)))

    # likelihood = tf.function(get_log_prob_fn(forward_model, true_observation, batch_dim=None))

    # points, L = nested_sample(likelihood, n_live, n_dims, n_repeats, n_compress)
    points, L = nested_sample(likelihood, n_live, n_dims, n_repeats)
    posterior, logw = samples(points, L)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    # likelihoods, samples = sess.run([likelihoods, samples])
    posterior, logw = sess.run([posterior, logw])

    print(posterior)
    print(posterior.shape)
    exit()
    
    # print(samples)
    # print(samples.shape)
    # plt.hist(samples.flatten())
    # plt.tight_layout()
    # plt.savefig('temp_nested_samples.png')
    # print(samples.numpy())  # should all be spread near 0.5