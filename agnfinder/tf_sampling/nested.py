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
        log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation, self.problem.fixed_params, self.problem.uncertainty)
        # log_prob_fn_w_args = lambda cube, *args: log_prob_fn(x)

        # parameters = 
        n_params = len(self.problem.true_params)

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
        sampler = dynesty.NestedSampler(log_prob_fn, dummy_prior, n_params)
        sampler.run_nested()
        result = sampler.results
        samples = result.samples
        # "Dynamic" nested sampling.
        # dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim)
        # dsampler.run_nested()
        # dresults = dsampler.results


        is_successful = np.ones(len(samples))  # for now
        metadata = {}
        return samples, is_successful, metadata

def dummy_prior(cube):
    return cube  # already a unit hypercube w/ flat priors
