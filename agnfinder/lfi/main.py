import os
import logging
import json
import argparse
from collections import OrderedDict

import h5py
import numpy as np
import pydelfi.priors as priors
from agnfinder import simulation_samples, simulation_utils

from agnfinder.lfi import train


TEST_SIM_LOC = 'data/lfi_test_case.json'

def load_simulator(emulate_ssp, noise):
    linear_simulator, _ = simulation_samples.get_photometry_simulator(
        emulate_ssp=emulate_ssp,
        noise=noise)  # add Gaussian noise according to best-guess sigma (currently an underestimate, probably)
    def simulator(theta, seed=1, simulator_args=None, batch=1):
        if np.max(theta) > 1. or np.min(theta) < 0.:
            logging.warning('Called with unphysical theta {}'.format(theta))
            return np.ones(12) * np.nan
        limits = OrderedDict({
            'log_mass': [8, 12], 
            'dust2': [0.001, 13.8],
            'tage': [0., 2.],
            'tau': [.1, 30],
            'log_agn_mass': [-7, np.log10(15)],
            'agn_eb_v': [0., 0.5],
            'log_agn_torus_mass': [-7, np.log10(15)]
        })
        denormalised_theta = simulation_utils.denormalise_theta(theta, limits)
        return np.log10(linear_simulator(denormalised_theta))

    with open(TEST_SIM_LOC, 'r') as f:
        test_sim = json.load(f)
        true_observation = np.array(test_sim['true_observation'])
        true_params = np.array(test_sim['true_params'])

    
    # print('Sim', simulator(true_params))
    # print('True', true_observation)
    print('\n')
    print('These should be similar!')
    print(list(zip(simulator(true_params), true_observation)))
    # exit()
    lower, upper = get_unit_bounds(param_dim=len(true_params))
    prior = priors.Uniform(lower, upper)

    return train.DelfiProblem(
        true_observation=true_observation,
        true_params=true_params,
        simulator=simulator,
        lower=lower,
        upper=upper,
        prior=prior)


def load_simulations(data_loc):
    logging.warning('Using data loc {}'.format(data_loc))
    assert os.path.isfile(data_loc)
    with h5py.File(data_loc, 'r') as f:
        theta = f['samples']['normalised_theta'][...]
        # hacky extra normalisation here, not great TODO
        simulated_y = np.log10(f['samples']['simulated_y'][...])

    # test set observation
    test_index = int(len(theta) / 8.)
    logging.info('Testing simulated galaxy {}'.format(test_index))
    true_params = theta[test_index]
    true_observation = simulated_y[test_index]

    # training set
    sim_params = np.vstack([theta[:test_index], theta[test_index+1:]])
    sim_data = np.vstack(
        [simulated_y[:test_index], simulated_y[test_index+1:]])

    lower, upper = get_unit_bounds(sim_params.shape[1])
    prior = priors.Uniform(lower, upper)

    # TODO temporary - save these for use as active sampling test case also
    with open(TEST_SIM_LOC, 'w') as f:
        json.dump(
            {
                'true_params': list(true_params),
                'true_observation': list(true_observation)
            },
            f
        )

    return train.DelfiProblem(
        true_observation=true_observation,
        true_params=true_params,
        sim_data=sim_data,
        sim_params=sim_params,
        lower=lower,
        upper=upper,
        prior=prior)

def get_unit_bounds(param_dim):
    # prior is uniform in normalised hypercube space
    # will be log-uniform in real space for log params, but that's fine
    lower = np.zeros(param_dim)
    upper = np.ones(param_dim)
    return lower, upper

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train pydelfi')
    parser.add_argument('run_name', type=str)
    parser.add_argument('--simulate', default=False,
                        dest='simulate', action='store_true')
    parser.add_argument('--data-loc', default=None, dest='data_loc', type=str)
    parser.add_argument('--test', default=False,
                        dest='test', action='store_true')
    parser.add_argument('--emulate-ssp', default=False,
                        dest='emulate_ssp', action='store_true')
    args = parser.parse_args()

    model_size = 'smaller'  # TODO properly explore possibilities
    name = '{}'.format(args.run_name)  # option to add more things automatically
    output_dir = os.path.join('results/lfi', name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    logging.basicConfig(
        filename=os.path.join(output_dir, '{}.log'.format(name)),
        format='%(asctime)s %(message)s',
        level=logging.INFO)

    # construct problem for delfi. Encapsultes physics. This should be a class.
    if args.test:
        # I broke this recently, and should be proper test case anyway TODO
        raise NotImplementedError
    elif args.simulate:
        problem = load_simulator(emulate_ssp=args.emulate_ssp, noise=True)
    elif args.data_loc is not None:
        problem = load_simulations(args.data_loc)
    else:
        raise ValueError('Execution requires either a simulator (--simulate) or previous sims (--data-loc=/path)')

    train.solve(problem, output_dir)
