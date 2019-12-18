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

        logging.info('Initial state: \n')
        for state in initial_state.numpy():
            logging.info(['{:.2f}'.format(param) for param in state])
        if self.problem.true_params is not None:
            logging.info('True params: \n')
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
            not_quite_true_params = np.vstack([self.problem.true_params for n in range(self.n_chains)] + np.random.rand(self.n_chains, len(self.problem.true_params)) * 0.03).astype(np.float32)
            initial_state = tf.constant(not_quite_true_params)
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
        samples, is_accepted = hmc(
            log_prob_fn=log_prob_fn,
            initial_state=initial_state,
            num_results=self.n_samples,
            num_burnin_steps=self.n_burnin
        )
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
def hmc(log_prob_fn, initial_state, num_results=int(10e3), num_burnin_steps=int(1e3)):

    # Initialize the HMC transition kernel.
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_fn,
            num_leapfrog_steps=3,
            step_size=2,
            state_gradients_are_stopped=True),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    @tf.function
    def run_chain():
        # Run the chain (with burn-in).
        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

        is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
        return samples, is_accepted

    samples, is_accepted = run_chain()

    return samples, is_accepted


def optimised_start(forward_model, observation, param_dim, n_chains, steps, initial_guess=None):
    start_time = datetime.datetime.now()
    n_best_params = int(n_chains / 3)
    assert n_best_params  # mustn't be 0
     # get n_chains of optimised parameter vectors, for speed
    all_best_params = find_best_params(forward_model, observation, param_dim, n_chains, steps)
    # keep only the best
    best_params = keep_top_params(all_best_params, forward_model, observation, n_chains, n_best_params)
    # randomly copy the best as needed to get back to n_chains starting points (all of which are now optimised)
    chosen_params = tf.gather(best_params, tf.random.uniform(minval=0, maxval=n_best_params-1, dtype=tf.int32, shape=[n_chains]))
    end_time = datetime.datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    logging.info('Done {} optimisations in {} seconds.'.format(n_chains, elapsed))
    logging.info('Picked {} starts from {} best params'.format(n_chains, n_best_params))
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


def many_random_starts(forward_model, observation, param_dim, n_chains):
    overproposal_factor = 10
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
