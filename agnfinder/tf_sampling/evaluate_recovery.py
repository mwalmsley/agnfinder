import os

import numpy as np
import tensorflow as tf  # just for eager toggle

from agnfinder.tf_sampling import deep_emulator, api, hmc

if __name__ == '__main__':
    
    tf.enable_eager_execution() 

    checkpoint_loc = 'results/checkpoints/weights_only/latest_tf'  # must match saved checkpoint of emulator
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    _, _, x_test, y_test = deep_emulator.data()
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    n_burnin = 1000
    n_samples = 2000
    n_chains = 128
    init_method = 'optimised'

    n_galaxies_to_check = 5
    all_true_params = np.zeros((n_galaxies_to_check, 7))
    all_best_estimates = np.zeros_like(all_true_params)
    for i in range(n_galaxies_to_check):
        true_params = x_test[i]
        true_observation = y_test[i]
        problem = api.SamplingProblem(true_observation, true_params, forward_model=emulator)
        sampler = hmc.SamplerHMC(problem, n_burnin, n_samples, n_chains, init_method=init_method)
        flat_samples = sampler()
        best_estimate = np.median(flat_samples, axis=0)

        all_true_params[i] = true_params
        all_best_estimates[i] = best_estimate  # ignore linting error

    save_dir = 'results/recovery/latest_{}'.format(init_method)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    np.savetxt(os.path.join(save_dir, 'true_params.txt'), all_true_params)
    np.savetxt(os.path.join(save_dir, 'best_estimates.txt'), all_best_estimates)
