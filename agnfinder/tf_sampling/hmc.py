import logging
import datetime
from typing import List
from sklearn.covariance import ShrunkCovariance

# from scipy import linalg
from scipy.stats import entropy
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from agnfinder.tf_sampling.api import Sampler, SamplingProblem, get_log_prob_fn
from agnfinder.tf_sampling import initial_starts


class SamplerHMC(Sampler):

    def __init__(self, problem: SamplingProblem, n_burnin, n_samples, init_method='random'):
        assert tf.executing_eagerly()  # required for sampling
        self.problem = problem
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        assert init_method in {'random', 'optimised', 'correct', 'roughly_correct'}
        self.init_method = init_method

    @property
    def n_chains(self):
        return len(self.problem.true_observation)

    def sample(self):
        start_time = datetime.datetime.now()

        initial_state = self.get_initial_state()  # numpy

        with np.printoptions(precision=2, suppress=False):
            logging.info('\nInitial state: ')
            logging.info(initial_state)
            logging.info('Median initial state:')
            logging.info(np.median(initial_state, axis=0))
            if self.problem.true_params is not None:
                logging.info('True params:')
                logging.info(self.problem.true_params)

        # initial run to find good MEAN step size, and reject chains which chains are not adapting
        initial_samples, is_accepted, step_sizes = self.run_hmc(
            initial_state,
            thinning=1,
            burnin_only=True)
        assert not np.any(np.isnan(initial_samples))

        # identify which chains have low acceptance
        accepted_per_galaxy = tf.reduce_mean(input_tensor=is_accepted, axis=0)
        good_acceptance = accepted_per_galaxy > tf.ones([self.n_chains]) * .6  # min acceptance of 60%

        for n, adapted in enumerate(good_acceptance.numpy()):
            if not adapted:
                logging.warning('Removing galaxy {} due to low acceptance (p={:.2f})'.format(n, accepted_per_galaxy[n]))

        # filter samples, true_observation (and true_params) to remove them
        initial_samples_filtered = tf.boolean_mask(
            tensor=initial_samples,
            mask=good_acceptance,
            axis=1
        )
        self.problem.filter_by_mask(good_acceptance)  # inplace
        # and also filter the step sizes
        step_sizes = step_sizes[good_acceptance]
        logging.info(f'Initial burn-in complete. Step sizes: {np.around(step_sizes, 5)}')

        # now do some normal samples at fixed step size, to estimate the std devs (i.e. the diagonal metric)
        logging.info('Sampling at fixed step size, to estimate diagonal metric')
        metric_samples, is_accepted, _ = self.run_hmc(initial_samples_filtered[-1], initial_step_sizes=step_sizes, thinning=1, find_metric=True)
        
        # std_devs = metric_samples.std(axis=0)
        # logging.info(f'Estimated std devs: {np.around(std_devs, 3)}')
        # std_devs = std_devs * np.median(step_sizes) / np.median(std_devs)  # scale to keep the same mean-ish (will burn-in again anyway)
        # std_devs_df = pd.DataFrame(data=std_devs, index=self.problem.observation_ids).reset_index()  # columns are params (int range)
        # std_devs_by_galaxy_df = std_devs_df.groupby('index').agg('mean')
        # std_devs_by_galaxy_lookup = dict(zip(std_devs_by_galaxy_df.index, std_devs_by_galaxy_df.values))
        # per_variable_initial_step_sizes = np.array([std_devs_by_galaxy_lookup[obs_id] for obs_id in self.problem.observation_ids])
        # logging.info(f'Per variable initial step sizes: {np.around(per_variable_initial_step_sizes, 5)}')
        
        bijector = get_decorrelation_bijector(metric_samples, self.problem.observation_ids)
        
        # scale current step size by std devs (will then adapt, keeping these ratios)
        # covariances, inv_covariances = get_covariances(metric_samples)  # force symmetric
        # print('covariance 0')
        # print(covariances[0])
        # np.savetxt('covariance_0.npy', covariances[0])
        # print('covariace 1')
        # print(covariances[1])
        # proposal_distributions = get_proposal_distributions(tf.constant(inv_covariances, dtype=tf.float32))
        # print('proposal shape')/
        # print(proposal_distributions.sample().shape)
        # continue, for real this time

        logging.info('Beginning final production burn-in/sampling')
        final_samples_metric, is_accepted, final_step_sizes = self.run_hmc(
            initial_samples_filtered[-1], 
            # initial_step_sizes=per_variable_initial_step_sizes,
            initial_step_sizes=step_sizes * 5,  # expect to be substantially larger than before, in new metric
            thinning=1,
            bijector=bijector
        )  
        logging.info(f'Final step sizes used: {np.around(final_step_sizes, 5)}')
        # note the thinning
        # print('final_samples before matmul')
        # print(final_samples_metric)
        # problem object is modified inplace to filter out failures
        # this is in hmc-metric space, still needs to be transformed back to parameter space

        # convert back out of metric space
        # note the matmul with coveriances, not inverse covariances
        # modify by column to preserve memory
        # for chain_n in tqdm.tqdm(range(final_samples_metric.shape[1])):
            # final_samples_metric[:, chain_n] = np.matmul(final_samples_metric[:, chain_n], np.sqrt(covariances[chain_n]))
        final_samples = final_samples_metric  # rename
        assert final_samples.shape[1] == np.sum(good_acceptance)  # auto-numpy cast?
        assert final_samples.shape[2] == self.problem.true_params.shape[1]

        # TODO am I supposed to be filtering for accepted samples? is_accepted has the same shape as samples, and is binary.
        end_time = datetime.datetime.now()
        logging.info('Total time for galaxies: {}s'.format( (end_time - start_time).total_seconds()))

        # for convenience, group samples by galaxy
        unique_ids = pd.unique(self.problem.observation_ids)
        if len(unique_ids) > 1:
            partition_key = dict(zip(unique_ids, range(len(unique_ids))))
            partitions = [partition_key[x] for x in self.problem.observation_ids]

            partitions = [partition_key[x] for x in self.problem.observation_ids]
            logging.info(f'Partitioning galaxies with {partition_key} with partitions {partitions}')
            # list of (sample, chain, param) samples, by galaxy
            # TODO could do this in numpy, as final_samples is already np.array, but whatever
            with tf.device('/CPU:0'):  # hard to fit this in memory on GPU
                samples_by_galaxy_chain_first = tf.dynamic_partition(
                    data=tf.transpose(final_samples, [1, 0, 2]),  # put chain index first temporarily
                    partitions=partitions,
                    num_partitions=len(unique_ids)
                )
                # put chain index back in second dim, and finally cast back to numpy
                samples_by_galaxy = [tf.transpose(x, [1, 0, 2]).numpy() for x in samples_by_galaxy_chain_first]
        else:  # only one anyway, so just send straight to list
            logging.warning(f'Only one unique observation found ({unique_ids}) - skipping partitioning')
            samples_by_galaxy = [final_samples]

        # list of metadata dicts, by galaxy
        metadata = [{} for n in range(len(samples_by_galaxy))]
        # metadata = [{'is_accepted': is_accepted[] for n in range(len(is_accepted))}] TODO not quite right, need to use partitions again
        # list of sample weights (here, all equal) by galaxy
        sample_weights = [np.ones((x.shape[:2])) for x in samples_by_galaxy]  # 0 and 1 dimensions
        # list of log evidence (here, not relevant - to remove?), by galaxy
        log_evidence = [np.ones_like(x) for x in sample_weights]
        return unique_ids, samples_by_galaxy, sample_weights, log_evidence, metadata



    def get_initial_state(self):
        if self.init_method == 'correct':
            assert self.problem.true_params is not None
            initial_state = tf.constant(self.problem.true_params, dtype=tf.float32)
        elif self.init_method == 'roughly_correct':
            assert self.problem.true_params is not None
            initial_state = initial_starts.roughly_correct_start(self.problem.true_params, self.n_chains)
        elif self.init_method == 'random':
            initial_state = tf.random.uniform(self.problem.true_params.shape, minval=0., maxval=1.)
            # initial_state = many_random_starts(self.problem.forward_model, self.problem.true_observation, self.problem.param_dim, self.n_chains)
        elif self.init_method == 'optimised':
            initial_chains = self.n_chains
            initial_state_unfiltered, _, is_successful = initial_starts.optimised_start(
                self.problem.forward_model,
                tf.constant(self.problem.true_observation),
                tf.constant(self.problem.fixed_params),
                tf.constant(self.problem.uncertainty),
                tf.constant(self.problem.param_dim),
                tf.constant(self.n_chains),
                steps=tf.constant(3000)
            )
            initial_state = tf.boolean_mask(
                initial_state_unfiltered,
                is_successful,
                axis=0
            )
            self.problem.filter_by_mask(is_successful)  # inplace
            logging.info(f'{is_successful.sum()} of {initial_chains} chains successful')
        else:
            raise ValueError('Initialisation method {} not recognised'.format(self.init_method))
        return initial_state

# remove tfp.bijectors.Sigmoid(low=0., high=1.)
    def run_hmc(self, initial_state, initial_step_sizes=None, thinning=1, bijector=tfp.bijectors.Identity(), burnin_only=False, find_metric=False):

        if burnin_only:
            # assert proposal_distributions is None
            # run burn-in, and then a few samples to measure adaption
            n_burnin = int(self.n_burnin / 2.)
            n_samples = 1000  # just enough to measure adaption
            assert thinning == 1
        elif find_metric:
            # assert proposal_distributions is None
            # with fixed step size (no burn-in), collect samples in euclidean metric
            n_samples = 5000  # 5000 about enough
            n_burnin = int(self.n_burnin / 10.)  # still apparently needs some burnin??
        else:  # continue an ongoing run
            n_samples = self.n_samples
            n_burnin = int(self.n_burnin  / 2.) # to update step size
        
        log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation, self.problem.fixed_params, self.problem.uncertainty)

        logging.info('Ready to go - beginning sampling at {}'.format(datetime.datetime.now().ctime()))
        start_time = datetime.datetime.now()
        samples, trace = hmc(
            log_prob_fn=log_prob_fn,
            initial_state=initial_state,
            initial_step_sizes=initial_step_sizes,
            n_samples=n_samples,  # before stopping and checking that adaption has succeeded
            n_burnin=n_burnin,
            thin=thinning,
            bijector=bijector
        )
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        samples = samples.numpy()
        ms_per_sample = 1000 * elapsed.total_seconds() / np.prod(samples.shape)  # not counting burn-in as a sample, so really quicker
        logging.info('Sampling {} x {} chains complete in {}, {:.3f} ms per sample'.format(n_samples, self.n_chains, elapsed, ms_per_sample))
        
        is_accepted = tf.cast(trace['is_accepted'], dtype=tf.float32)
        record_acceptance(is_accepted.numpy())

        final_step_sizes = tf.cast(trace['step_size'][-1], dtype=tf.float32)

        return samples, is_accepted, final_step_sizes


def get_covariances(samples):
    estimators = [ShrunkCovariance(assume_centered=False).fit(samples[:, n]) for n in range(samples.shape[1])]  
    covariances = [x.covariance_ for x in estimators]
    inv_covariances = [x.precision_ for x in estimators]
    # by chain (could average across galaxies?)
    # covariances = [np.maximum(x, x.transpose()) for x in raw_covariances]  # force symmetric, may not be needed
    # raw_inv_covariances = [linalg.inv(x) for x in raw_covariances]
    # inv_covariances = [np.maximum(x, x.transpose()) for x in raw_inv_covariances]  # force symmetric, may not be needed
    return covariances, inv_covariances


def get_proposal_distributions(inv_covariance_matrices):
    # will use one distribution with parameters along batch dim, instead of a list of distributions
    mu = tf.zeros(len(inv_covariance_matrices[0]))  # length of each cov matrix = num params
    mus = [mu for _ in inv_covariance_matrices]
    scale_trils = [tf.linalg.cholesky(tf.cast(x, dtype=tf.float32)) for x in inv_covariance_matrices]
    return tfp.distributions.MultivariateNormalTriL(
        loc=mus,
        scale_tril=scale_trils
    )

def get_kl_vs_other_chains(chains: np.ndarray, min_p=1e-5):
    n_chains = chains.shape[1]
    param_n = 0
    kl_divs = np.zeros(n_chains) * np.nan
    bins = np.linspace(chains[:, :, param_n].min(), chains[:, :, param_n].max(), 10)
    for chain_n in range(n_chains):
        mask = np.ones(n_chains).astype(bool)
        mask[chain_n] = False
        this_chain = chains[:, chain_n]
        other_chains = chains  # actually all chains, to ensure support
        other_probs, _ = np.histogram(other_chains[:, :, param_n], density=True, bins=bins)
        this_probs, _ = np.histogram(this_chain[:, param_n], density=True, bins=bins)
        allowed_probs = (other_probs > min_p) & (this_probs > min_p)
        kl_divs[chain_n] = entropy(other_probs[allowed_probs], this_probs[allowed_probs])  # calculates KL of q (second arg) approximating p (first arg)
    if kl_divs.mean() < 1e-6:
        raise ValueError('KL div calculation failed')
    return kl_divs

def chains_pass_quality_check(chains):
    kl_divs = get_kl_vs_other_chains(chains)
    kl_is_successful = kl_divs < (np.median(kl_divs) + kl_divs.std() * 2)

    typical_std = np.median(chains.std(axis=0)[:, 0])
    chain_stds = chains.std(axis=0)[:, 0]
    std_is_successful = chain_stds > typical_std * 0.1

    chain_is_successful = kl_is_successful & std_is_successful
    return chain_is_successful

# don't decorate with tf.function
def hmc(log_prob_fn, initial_state, initial_step_sizes=None, n_samples=int(10e3), n_burnin=int(1e3), thin=1, bijector=tfp.bijectors.Identity()):

    assert len(initial_state.shape) == 2  # should be (chain, variables)

    if initial_step_sizes is None:
        initial_step_size = .005 # starting point, will be updated by step size adaption. Roughly, the median per-variable std
        initial_step_sizes = tf.fill(initial_state.shape, initial_step_size)  # each chain (and variable) will have own step size
    # this is crucial now that each chain is potentially a different observation

    # NUTS
    # step size adaption from
    # https://github.com/tensorflow/probability/issues/549
    # transition_kernel = tfp.mcmc.NoUTurnSampler(
    #     target_log_prob_fn=log_prob_fn,
    #     step_size=initial_step_sizes
    # )
    # adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #     transition_kernel,
    #     num_adaptation_steps=int(n_burnin * 0.8),
    #     step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
    #     step_size_getter_fn=lambda pkr: pkr.step_size,
    #     log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio
    # )

    # or HMC
    transition_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=initial_step_sizes,
        # L should be large enough that there isn't high autocorrelation between samples
        num_leapfrog_steps=10,  # https://stats.stackexchange.com/questions/304942/how-to-set-step-size-in-hamiltonian-monte-carlo
        state_gradients_are_stopped=True,  # probably not important as we're not doing any optim. with the samples,
        # proposal_distributions=proposal_distributions
    )

    # scale param is NOT on unit cube, need to turn off softmax for now
    # use softmax to move from constrained hypercube (unit) space to unconstrained space,
    # which is easier to sample near the hypercube boundaries
    # constrain_to_unit_bijector =   # these are default, but let's be explicit
    transformed_transition_kernel = tfp.mcmc.TransformedTransitionKernel(
        transition_kernel,
        bijector # will broadcast elementwise to all chains/params, I think
    )

    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        transformed_transition_kernel,
        num_adaptation_steps=int(n_burnin * 0.8),
    )

    # seperate function so it can be decorated
    @tf.function(experimental_compile=True)
    def run_chain():
        # Run the chain (with burn-in).
        # chain_output = tfp.mcmc.sample_chain(
        #     num_results=n_samples,
        #     num_burnin_steps=n_burnin,
        #     current_state=initial_state,
        #     kernel=adaptive_kernel,
        #     # https://github.com/tensorflow/probability/blob/f90448698cc2a16e20939686ef0d5005aad95f29/tensorflow_probability/python/mcmc/nuts.py#L72
        #     trace_fn=lambda _, prev_kernel_results: {'is_accepted': prev_kernel_results.inner_results.is_accepted, 'step_size': prev_kernel_results.inner_results.step_size}
        # )
        chain_output = tfp.mcmc.sample_chain(
            num_results=n_samples,
            num_burnin_steps=n_burnin,
            current_state=initial_state,
            kernel=adaptive_kernel,
            parallel_iterations=1,  # makes no difference at all to performance, when tested - worth checking before final run
            # https://github.com/tensorflow/probability/blob/f90448698cc2a16e20939686ef0d5005aad95f29/tensorflow_probability/python/mcmc/nuts.py#L72
            # this is pretty awful, but it works. adaptive kernel < transformed kernel < hmc kernel
            trace_fn=lambda _, prev_kernel_results: {
                'is_accepted': prev_kernel_results.inner_results.inner_results.is_accepted,
                'step_size': prev_kernel_results.inner_results.inner_results.accepted_results.step_size
                },
            num_steps_between_results=thin  # thinning factor, to run much longer with reasonable memory
        )

        samples, trace = chain_output
        return samples, trace

    samples, trace = run_chain()

    return samples, trace

def record_acceptance(is_accepted):
    logging.info('Mean acceptance ratio over all chains: {:.4f}'.format(np.mean(is_accepted)))
    mean_acceptance_per_chain = np.mean(is_accepted, axis=0)
    for chain_i, chain_acceptance in enumerate(mean_acceptance_per_chain):
        if chain_acceptance < 0.01:
            logging.critical(f'HMC failed to adapt for chain {chain_i} - is step size too small? Is there some burn-in?')
        elif chain_acceptance < 0.3:
            logging.critical('Acceptance ratio is low for chain {}: ratio {:.2f}'.format(chain_i, chain_acceptance))


def get_decorrelation_bijector(samples, observation_ids):
    # flattens across chains for same problem
    chols_per_obs = []
    unique_observation_ids = set(observation_ids)
    logging.info(f'Unique observations: {unique_observation_ids}')
    for obs_id in unique_observation_ids:  # unique only
        mask = np.array([x == obs_id for x in observation_ids], dtype=bool)
        assert mask.sum() > 0
        obs_samples = samples[:, mask]
        logging.info(f'Found {obs_samples.shape} samples for obs {obs_id}')
        # filter bad chains
        chains_are_good = chains_pass_quality_check(obs_samples)
        logging.info(f'Obs {obs_id}: {chains_are_good.sum()} chains of {len(chains_are_good)} pass quality check')
        good_obs_samples = obs_samples[:, chains_are_good]

        flattened_obs_samples = good_obs_samples.reshape(-1, samples.shape[2])
        cov = tfp.stats.covariance(flattened_obs_samples)  # or just np cov?
        logging.info(f'estimated covariance for observation {obs_id}: {cov}')
        np.savetxt('latest_cov.npy', cov)  # temporary
        chol = tf.linalg.cholesky(cov)
        # logging.info(f'estimated chol for observation {obs_id}: {chol}')
        chols_per_obs.append(chol)

    chols_by_chain_lookup = dict(zip(unique_observation_ids, chols_per_obs))
    chols_by_chain = [chols_by_chain_lookup[obs_id] for obs_id in observation_ids]
    bijector = tfp.bijectors.ScaleMatvecTriL(scale_tril=chols_by_chain)  # different dists over batch dim
    return bijector
