import logging
import datetime

import numpy as np
import tensorflow as tf
import emcee
import tqdm

from agnfinder.tf_sampling.api import Sampler, get_log_prob_fn, get_log_prob_fn_variable_batch, SamplingProblem
from agnfinder.tf_sampling import initial_starts


class SamplerEmcee(Sampler):

    def __init__(self, problem: SamplingProblem, n_burnin, n_samples, n_chains, init_method='optimised'):
        assert tf.executing_eagerly()  # required for sampling
        self.problem = problem
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        self.n_chains = n_chains
        assert init_method == 'optimised'  # only support this
        self.init_method = init_method


    def sample(self):
        is_successful = np.zeros(len(self.problem.true_observation)).astype(bool)
        sample_list = []

        # run serially
        for galaxy_index in tqdm.tqdm(range(len(self.problem.true_observation)), unit=' galaxies'):

            # pull out values for this galaxy
            true_observation_g = self.problem.true_observation[galaxy_index]
            fixed_params_g = self.problem.fixed_params[galaxy_index]
            uncertainty_g = self.problem.uncertainty[galaxy_index]

            n_params = self.problem.param_dim
            nwalkers = 256  # 512 on zeus?

            true_observation_walkers = tf.constant(np.stack([true_observation_g for _ in range(nwalkers)], axis=0))
            fixed_params_walkers = tf.constant(np.stack([fixed_params_g for _ in range(nwalkers)], axis=0))
            uncertainty_walkers = tf.constant(np.stack([uncertainty_g for _ in range(nwalkers)], axis=0))

            # emcee wants initial start *with* a batch dim, where batch=walker rather than batch=galaxy. Can re-use the same code.
            p0_unfiltered, is_successful = initial_starts.optimised_start(
                self.problem.forward_model,
                true_observation_walkers,
                fixed_params_walkers,
                uncertainty_walkers,
                self.problem.param_dim,
                nwalkers,  # not n_chains=num galaxies, walkers as batch dim!
                steps=3000,
                n_attempts=5
            )

            logging.info(f'{is_successful.sum()} of {self.n_chains} chains successful')

            p0 = p0_unfiltered[is_successful]
            self.problem.filter_by_mask(is_successful)  # inplace
            self.n_chains = tf.reduce_sum(input_tensor=tf.cast(is_successful, tf.int32)) # n_chains doesn't do anything at this point, I think

            # emcee log prob must be able to handle variable batch dimension, for walker subsensembles (here, actually a hassle)
            log_prob_fn = get_log_prob_fn_variable_batch(
                self.problem.forward_model, # forward model requires a batch dim, set to 1
                tf.constant(true_observation_g, dtype=tf.float32),
                tf.constant(fixed_params_g, dtype=tf.float32),
                tf.constant(uncertainty_g, dtype=tf.float32)
            )

            def temp_log_prob_fn(x):
                result = log_prob_fn(tf.constant(x, dtype=tf.float32)).numpy()
                result[result<-1e9] = -np.inf
                # result[np.isnan(result)] = -np.inf
                return result

            sampler = emcee.EnsembleSampler(nwalkers, n_params, temp_log_prob_fn, vectorize=True)  # x will be list of position vectors
            
            start_time = datetime.datetime.now()
            logging.info(f'Begin sampling at {start_time}')

            state = sampler.run_mcmc(p0, self.n_burnin, progress=True)
            logging.info('Burn-in complete')
            logging.info(f'Acceptance: {sampler.acceptance_fraction.mean()} +/- {2*sampler.acceptance_fraction.std()}')
            sampler.reset()


            sampler.run_mcmc(state, self.n_samples, progress=True, thin_by=10)
            time_elapsed = datetime.datetime.now() - start_time
            seconds_per_sample = time_elapsed.seconds / (self.n_samples * nwalkers)
            logging.info(f'emcee sampling complete in {time_elapsed}, {seconds_per_sample}s per sample')
            acceptance = sampler.acceptance_fraction
            logging.info(f'Acceptance: {acceptance.mean()} +/- {2*acceptance.std()}')

            sample_list.append(sampler.get_chain(flat=True))  # flattened
            is_successful[galaxy_index] = True  # for now

        num_samples_by_galaxy = [len(x) for x in sample_list]
        max_samples = np.max(num_samples_by_galaxy)
        samples = np.zeros((max_samples, len(sample_list), n_params))
        sample_weights = np.ones((max_samples, len(sample_list)))
        log_evidence = np.zeros((max_samples, len(sample_list)))
        for n, x in enumerate(sample_list):
            samples[:len(x), n, :] = x  # sample, galaxy, param
        metadata = {}
        return samples, is_successful, sample_weights, log_evidence, metadata
