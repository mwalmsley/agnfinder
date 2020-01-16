import tensorflow as tf

from nestorflow.nested import nested_sample

from agnfinder.tf_sampling.api import Sampler, get_log_prob_fn


class SamplerNested(Sampler):

    def __init__(self, problem, n_live):
        self.problem = problem
        self.n_live = n_live
        # self.n_repeats = problem.n_dim * 2
        self.n_repeats = 1

    def sample(self):
        log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation)
        points, L = nested_sample(
            log_prob_fn,
            self.n_live,
            self.problem.n_dim,
            self.n_repeats
        )
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        likelihoods, samples = sess.run([L, points])
        return samples
