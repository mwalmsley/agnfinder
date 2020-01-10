import logging
import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from agnfinder.tf_sampling.api import Sampler, SamplingProblem, get_log_prob_fn


class SamplerHMC(Sampler):

    def __init__(self, problem: SamplingProblem, n_burnin, n_samples, n_chains, init_method='random'):
        assert tf.executing_eagerly()  # required for sampling
        self.problem = problem
        self.n_burnin = n_burnin
        self.n_samples = n_samples
        self.n_chains = n_chains
        assert init_method in {'random', 'optimised', 'correct', 'roughly_correct'}
        self.init_method = init_method

    def sample(self):
        start_time = datetime.datetime.now()

        log_prob_fn = get_log_prob_fn(self.problem.forward_model, self.problem.true_observation, batch_dim=self.n_chains)
        initial_state = self.get_initial_state()

        logging.info('\nInitial state:')
        for state in initial_state.numpy():
            logging.info(['{:.2f}'.format(param) for param in state])
        logging.info('\nMedian initial state:')
        logging.info(['{:.2f}'.format(param) for param in np.median(initial_state, axis=0)])
        if self.problem.true_params is not None:
            logging.info('\nTrue params:')
            logging.info(['{:.2f}'.format(param) for param in self.problem.true_params])

        samples = self.run_hmc(log_prob_fn, initial_state)

        # print('True vs. median recovered parameters: ', list(zip(self.problem.true_params, np.median(samples, axis=1))))
        end_time = datetime.datetime.now()
        logging.info('Total time for galaxy: {}s'.format( (end_time - start_time).total_seconds()))
        return samples

    def get_initial_state(self):
        if self.init_method == 'correct':
            assert self.problem.true_params is not None
            true_params_stacked = np.vstack([self.problem.true_params.astype(np.float32) for n in range(self.n_chains)])
            initial_state = tf.constant(true_params_stacked)
        elif self.init_method == 'roughly_correct':
            assert self.problem.true_params is not None
            initial_state = roughly_correct_start(self.problem.true_params, self.n_chains)
        elif self.init_method == 'random':
            initial_state = many_random_starts(self.problem.forward_model, self.problem.true_observation, self.problem.n_dim, self.n_chains)
        elif self.init_method == 'optimised':
            initial_state = optimised_start(self.problem.forward_model, self.problem.true_observation, self.problem.n_dim, self.n_chains, steps=3000)
        else:
            raise ValueError('Initialisation method {} not recognised'.format(self.init_method))
        return initial_state

    def run_hmc(self, log_prob_fn, initial_state):
        logging.info('Ready to go - beginning sampling at {}'.format(datetime.datetime.now().ctime()))
        start_time = datetime.datetime.now()
        samples, trace = hmc(
            log_prob_fn=log_prob_fn,
            initial_state=initial_state,
            n_samples=self.n_samples,
            n_burnin=self.n_burnin
        )
        is_accepted = tf.reduce_mean(tf.cast(trace['is_accepted'], dtype=tf.float32))
        samples = samples.numpy()
        is_accepted = is_accepted.numpy()
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        ms_per_sample = 1000 * elapsed.total_seconds() / (self.n_samples * self.n_chains)  # not counting burn-in as a sample, so really quicker
        print('Sampling {} x {} chains complete in {}, {:.3f} ms per sample'.format(self.n_samples, self.n_chains, elapsed, ms_per_sample))
        
        print('Acceptance ratio: {:.4f}'.format(is_accepted))
        if is_accepted < 0.01:
            logging.critical('HMC failed to adapt - is step size too small? Is there some burn-in?')
        elif is_accepted < 0.3:
            print('Warning - acceptance ratio is low!')
        return samples

# don't decorate
def hmc(log_prob_fn, initial_state, n_samples=int(10e3), n_burnin=int(1e3)):

    assert len(initial_state.shape) == 2  # should be (chain, variables)

    # NUTS
    initial_step_size = 1.  # starting point, will be updated by step size adaption
    initial_step_sizes = tf.fill(initial_state.shape, initial_step_size)  # each chain will have own step size
    transition_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=initial_step_sizes
    )
    transition_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=initial_step_sizes,
        num_leapfrog_steps=100
    )
    # step size adaption
    # https://github.com/tensorflow/probability/issues/549
    # adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #     transition_kernel,
    #     num_adaptation_steps=int(n_burnin * 0.8),
    #     step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
    #     step_size_getter_fn=lambda pkr: pkr.step_size,
    #     log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio
    # )
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        transition_kernel,
        num_adaptation_steps=int(n_burnin * 0.8),
    )


    # seperate function so it can be decorated
    @tf.function
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
            trace_fn=lambda _, prev_kernel_results: {'is_accepted': prev_kernel_results.inner_results.is_accepted}
        )

        samples, trace = chain_output
        return samples, trace

    samples, trace = run_chain()

    return samples, trace


def optimised_start(forward_model, observation, param_dim, n_chains, steps, initial_guess=None):
    start_time = datetime.datetime.now()
    # n_best_params = int(n_chains / 3)
    # assert n_best_params  # mustn't be 0
     # get n_chains of optimised parameter vectors, for speed
    all_best_params = find_best_params(forward_model, observation, param_dim, n_chains, steps)
    # keep only the best
    # best_params = keep_top_params(all_best_params, forward_model, observation, n_chains, n_best_params)
    # randomly copy the best as needed to get back to n_chains starting points (all of which are now optimised)
    # chosen_params = tf.gather(best_params, tf.random.uniform(minval=0, maxval=n_best_params-1, dtype=tf.int32, shape=[n_chains]))
    chosen_params = tf.reshape(all_best_params, [n_chains, param_dim])
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    logging.info('Done {} optimisations in {} seconds.'.format(n_chains, elapsed))
    # logging.info('Picked {} starts from {} best params'.format(n_chains, n_best_params))
    return chosen_params

def find_best_params(forward_model, observation, param_dim, batch_dim, steps, initial_guess=None):
    log_prob_fn = get_log_prob_fn(forward_model, observation, batch_dim=batch_dim)
    if initial_guess is not None:
        params = tf.Variable(tf.stack([initial_guess for _ in range(batch_dim)], axis=0), dtype=tf.float32)
    else:
        params = tf.Variable(tf.random.uniform(shape=(batch_dim, param_dim), dtype=np.float32), dtype=tf.float32)
    best_params = find_minima(lambda x: -log_prob_fn(x),  params, steps)  # a very important minus sign...
    return best_params


# TODO convert to be tf.function() friendly, creating no ops
# TODO measure convergence and adjust steps, learning rate accordingly
def find_minima(func, initial_guess, steps=1000,  optimizer=tf.keras.optimizers.Adam()):
    # logging.info('Initial guess neglogprob: {}'.format(['{:4.1f}'.format(x) for x in func(initial_guess)]))
    for _ in range(steps):
        with tf.GradientTape() as tape:
            func_value = func(initial_guess)
            grads = tape.gradient(func_value, [initial_guess])[0]
            grads_and_vars = [(grads, initial_guess)]
            optimizer.apply_gradients(grads_and_vars)  # inplace
            tf.assign(initial_guess, tf.clip_by_value(initial_guess, 0.01, 0.99))
    # logging.info('Final neglogprob: {}'.format(['{:4.1f}'.format(x) for x in func(initial_guess)]))
    return initial_guess


def many_random_starts(forward_model, observation, param_dim, n_chains, overproposal_factor=None):
    if overproposal_factor is None:
        assert n_chains < 100
        overproposal_factor = int(10 - (7 * 0.01 * n_chains))  # down from 10. Customised for laptop memory TODO tweak for Glamdring?
    overproposed_initial_state = tf.random.uniform(shape=(n_chains * overproposal_factor, param_dim))
    initial_state = keep_top_params(
        overproposed_initial_state,
        forward_model,
        observation,
        n_chains * overproposal_factor,  # specified explicitly but might not need to
        n_chains)
    return initial_state

def keep_top_params(all_params, forward_model, true_observation, initial_dim, final_dim):
    log_prob_fn = get_log_prob_fn(forward_model, true_observation, batch_dim=initial_dim)
    initial_log_probs = log_prob_fn(all_params)
    initial_state = tf.gather(all_params, tf.argsort(initial_log_probs, direction='DESCENDING'))[:final_dim]
    return initial_state

def roughly_correct_start(true_params, n_chains):
    not_quite_true_params = np.vstack([true_params for n in range(n_chains)] + np.random.rand(n_chains, len(true_params)) * 0.03).astype(np.float32)
    initial_state = tf.constant(not_quite_true_params)
    return initial_state