import logging
import datetime
from typing import List

import pandas as pd
import numpy as np
import tensorflow as tf
import emcee
import tqdm

from agnfinder.tf_sampling.api import Sampler, get_log_prob_fn, get_log_prob_fn_variable_batch, SamplingProblem
from agnfinder.tf_sampling import initial_starts


class SamplerEmcee(Sampler):

    def __init__(self, problem: SamplingProblem, n_burnin: int, n_samples: int, init_method='optimised'):
        assert tf.executing_eagerly()  # required for sampling
        self.problem = problem
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        assert init_method == 'optimised'  # only support this
        self.init_method = init_method

    @property
    def n_observations(self):
        return len(self.problem.true_observation)

    def sample(self):
        sample_list = []
        is_successful_list = []
        metadata_list = []

        id_counts = pd.value_counts(self.problem.observation_ids)
        if id_counts.max() > 1:
            logging.warning('Some observations are repeated - emcee cannot parallelise observations, so this will be slower. Is this your intention?')
            logging.warning(id_counts[id_counts > 1])
        # run serially
        for galaxy_n, galaxy_name in tqdm.tqdm(enumerate(self.problem.observation_ids), unit=' galaxies'):
            logging.info(f'Running emcee on galaxy {galaxy_name}')

            # pull out values for this galaxy
            true_observation_g = self.problem.true_observation[galaxy_n]
            fixed_params_g = self.problem.fixed_params[galaxy_n]
            uncertainty_g = self.problem.uncertainty[galaxy_n]

            n_params = self.problem.param_dim
            # note that --n-chains is actually n-galaxies for emcee, should change
            nwalkers = 256  # 512 on zeus?

            # emcee wants initial start *with* a batch dim, where batch=walker rather than batch=galaxy. Can re-use the same code.
            n_parallel_attempts = 100
            true_observation_attempts = tf.constant(np.stack([true_observation_g for _ in range(n_parallel_attempts)], axis=0))
            fixed_params_attempts = tf.constant(np.stack([fixed_params_g for _ in range(n_parallel_attempts)], axis=0))
            uncertainty_attempts = tf.constant(np.stack([uncertainty_g for _ in range(n_parallel_attempts)], axis=0))
            p0_unfiltered, neg_log_p_unfiltered, good_initial_start = initial_starts.optimised_start(
                self.problem.forward_model,
                true_observation_attempts,
                fixed_params_attempts,
                uncertainty_attempts,
                self.problem.param_dim,
                n_parallel_attempts,  # separate initial start for every walker
                steps=3000,
                n_attempts=1  # all parallel attempts are for the same problem, so already plenty without needing to retry
            )
            # print('example starts: ', p0_unfiltered[:5])

            logging.info(f'{good_initial_start.sum()} of {n_parallel_attempts} chains successful')
            p0_filtered = p0_unfiltered[good_initial_start]
            neg_log_p = neg_log_p_unfiltered[good_initial_start]
            # print(neg_log_p)
    
            logging.info(f'Best identified start: {np.min(neg_log_p)}')
            best_p0_index = np.argmin(neg_log_p)
            best_p0 = p0_filtered[best_p0_index]
            logging.info(f'Starting walkers in ball around most likely theta: {best_p0}')
            ball_radius = 0.01
            best_p0_repeated = np.stack([best_p0 for _ in range(nwalkers)], axis=0)
            # print(best_p0_repeated.shape)
            p0_ball = best_p0_repeated + np.random.rand(*best_p0_repeated.shape) * ball_radius
            # print(p0_ball.shape)
            logging.info(p0_ball)
            # exit()

            # emcee log prob must be able to handle variable batch dimension, for walker subsensembles (here, actually a hassle)
            log_prob_fn = get_log_prob_fn_variable_batch(
                self.problem.forward_model, # forward model requires a batch dim, set to 1
                tf.constant(true_observation_g, dtype=tf.float32),
                tf.constant(fixed_params_g, dtype=tf.float32),
                tf.constant(uncertainty_g, dtype=tf.float32)
            )

            def temp_log_prob_fn(x):
                result = log_prob_fn(tf.constant(x, dtype=tf.float32)).numpy()
                result[result < -1e9] = -np.inf
                return result

            sampler = emcee.EnsembleSampler(nwalkers, n_params, temp_log_prob_fn, vectorize=True)  # x will be list of position vectors
            thinning = 1
            
            start_time = datetime.datetime.now()
            logging.info(f'Begin sampling at {start_time}')

            state = sampler.run_mcmc(p0_ball, self.n_burnin, thin_by=thinning, progress=True)
            logging.info('Burn-in complete')
            logging.info(f'Acceptance: {sampler.acceptance_fraction.mean()} +/- {2*sampler.acceptance_fraction.std()}')
            sampler.reset()
            sampler.run_mcmc(state, self.n_samples, thin_by=thinning, progress=True)
            time_elapsed = datetime.datetime.now() - start_time
            seconds_per_sample = time_elapsed.seconds / (self.n_samples * nwalkers)
            logging.info(f'emcee sampling complete in {time_elapsed}, {seconds_per_sample}s per sample')
            acceptance = sampler.acceptance_fraction
            logging.info(f'Acceptance: {acceptance.mean()} +/- {2*acceptance.std()}')
            metadata_list.append({'acceptance': acceptance.mean()})

            sample_list.append(sampler.get_chain(flat=False))  # not flattened
            # for now, make no attempt to filter for success of walkers
            # (already filtered for successful initial condition)
            is_successful_list.append(True)





        assert len(is_successful_list) == len(sample_list)
        is_successful = np.array(is_successful_list)
        self.problem.filter_by_mask(is_successful)  # inplace

        # copied from hmc.py
        unique_ids = pd.unique(self.problem.observation_ids)
        sample_weights = [np.ones((x.shape[:2])) for x in sample_list]  # 0 and 1 dimensions
        # list of log evidence (here, not relevant - to remove?), by galaxy
        log_evidence = [np.ones_like(x) for x in sample_weights]
        return unique_ids, sample_list, sample_weights, log_evidence, metadata_list
