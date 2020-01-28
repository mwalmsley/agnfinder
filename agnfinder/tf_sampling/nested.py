import numpy as np
import tensorflow as tf

# from nestorflow.nested import nested_sample
# from pymultinest.solve import solve
import dynesty

from agnfinder.tf_sampling.api import Sampler, get_log_prob_fn, SamplingProblem


class SamplerNested(Sampler):

    def __init__(self, problem, n_live):
        self.problem = problem
        self.n_live = n_live
        # self.n_repeats = problem.n_dim * 2
        self.n_repeats = 1

    def sample(self):
        is_successful = np.zeros(len(self.problem.true_observation))
        sample_list = []
        for galaxy_index in range(len(self.problem.true_observation)):
            log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation[:1], self.problem.fixed_params[:1], self.problem.uncertainty[:1])
            n_params = self.problem.true_params.shape[1]

            # run MultiNest
            # result = solve(
            #     LogLikelihood=myloglike,
            #     Prior=prior, 
            #     n_dims=n_params,
            #     n_live_points=problem.n_live,
            #     outputfiles_basename=output_dir,
            #     verbose=True
            # )
            # samples = result['samples']

            # run Dynesty
            # "Static" nested sampling.
            # sampler = dynesty.NestedSampler(
            #     lambda x: float(log_prob_fn(tf.expand_dims(tf.cast(x, tf.float32), axis=0)).numpy()),
            #     dummy_prior,
            #     n_params
            # )
            # "Dynamic" nested sampling.
            sampler = dynesty.DynamicNestedSampler(
                lambda x: float(log_prob_fn(tf.expand_dims(tf.cast(x, tf.float32), axis=0)).numpy()),
                dummy_prior,
                n_params
            )
            sampler.run_nested()
            result = sampler.results
            samples = result.samples
            sample_list.append(samples)

            is_successful[galaxy_index] = 1  # for now

        num_samples_by_galaxy = [len(x) for x in sample_list]
        max_samples = np.max(num_samples_by_galaxy)
        samples = np.zeros((max_samples, len(sample_list), n_params))
        for n, x in enumerate(sample_list):
            samples[:, n, :] = x  # batch/galaxy is dimension 1 of samples - I should perhaps change this
        metadata = {}
        return samples, is_successful, metadata

def dummy_prior(cube):
    return cube  # already a unit hypercube w/ flat priors
