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
        is_successful = np.zeros(len(self.problem.true_observation)).astype(bool)
        sample_list = []
        sample_weights_list = []
        log_evidence_list = []
        for galaxy_index in range(len(self.problem.true_observation)):
            log_prob_fn = get_log_prob_fn(
                self.problem.forward_model, 
                np.expand_dims(self.problem.true_observation[galaxy_index], axis=0),
                np.expand_dims(self.problem.fixed_params[galaxy_index], axis=0),
                np.expand_dims(self.problem.uncertainty[galaxy_index], axis=0)
            )
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
            # print(result)
            # print(result.keys())
            # samples_list.append(result['samples'])
            # # https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/solve.py#L80
            # sample_weights_list.append(np.ones(len(result['samples'])))  # doesn't seem to include this?
            # log_evidence_list.append(result['logZ'])


            # run Dynesty
            # "Static" nested sampling.
            sampler = dynesty.NestedSampler(
                lambda x: float(log_prob_fn(tf.expand_dims(tf.cast(x, tf.float32), axis=0)).numpy()),
                dummy_prior,
                n_params
            )
            # "Dynamic" nested sampling.
            # sampler = dynesty.DynamicNestedSampler(
            #     lambda x: float(log_prob_fn(tf.expand_dims(tf.cast(x, tf.float32), axis=0)).numpy()),
            #     dummy_prior,
            #     n_params
            # )
            sampler.run_nested()
            result = sampler.results
            sample_list.append(result.samples)
            sample_weights_list.append(result.logwt)
            log_evidence_list.append(result.logz)

            is_successful[galaxy_index] = True  # for now

        num_samples_by_galaxy = [len(x) for x in sample_list]
        max_samples = np.max(num_samples_by_galaxy)
        samples = np.zeros((max_samples, len(sample_list), n_params))
        sample_weights = np.zeros((max_samples, len(sample_list)))
        log_evidence = np.zeros((len(sample_list)))
        for n, x in enumerate(sample_list):
            samples[:len(x), n, :] = x  # sample, galaxy, param
        for n, x in enumerate(sample_weights_list):
            sample_weights[:len(x), n] = x  # sample, galaxy
        for n, x in enumerate(log_evidence_list):
            print(x)
            print(x.shape)
            log_evidence[n] = x  # galaxy
        metadata = {}
        return samples, is_successful, sample_weights, log_evidence, metadata

def dummy_prior(cube):
    return cube  # already a unit hypercube w/ flat priors
