from collections import OrderedDict
import json
import pickle
import datetime
import os
import shutil

import numpy as np
import pymultinest
from pymultinest.solve import solve

from agnfinder.models import InterpolatedModel
from agnfinder.inference import gaussian_loglikelihood


def scale(normed_param, bound_values):
    # convert from unit scale to physical scale via bound_values
    return normed_param * (bound_values[1] - bound_values[0]) + bound_values[0]


# not currently used - inverse of scale
def normalise(param, bound_values):
    return param - bound_values[0] / (bound_values[1] - bound_values[0])


def scale_parameters(normed_params, bounds):
    # prior function which maps from [0:1] to the parameter space
    # is this just a really bad name? How to include non-uniform priors?
    # TODO make this elegant
    params = np.zeros(len(normed_params))
    # print(normed_params, bounds)
    # params[0] = scale(normed_params[0], bounds=bounds['lambda'])
    # params[1] = scale(normed_params[1], bounds=bounds['eb_v'])
    # params[2] = scale(normed_params[2], bounds=bounds['z'])
    normed_param_bounds_pair = list(zip(normed_params, bounds.values()))
    for n, (normed_param, bound_values) in enumerate(normed_param_bounds_pair):
        params[n] = scale(normed_param, bound_values=bound_values)
    return params


def run(model_loc, output_dir, resume):

    model = pickle.load(open(model_loc, 'rb'))

    if not resume:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    observed_w1 = 25.5

    # run MultiNest
    start_time = datetime.datetime.now()
    result = solve(
        LogLikelihood=lambda x: gaussian_loglikelihood(model, observed_w1, x),
        Prior=lambda x: scale_parameters(x, model.bounds), 
        n_dims=len(model.parameters()),
        outputfiles_basename='out/',
        resume=resume,
        verbose=False,
        max_iter=0,
        importance_nested_sampling=True,
        sampling_efficiency='parameter'
    )
    end_time = datetime.datetime.now()
    print('time', end_time - start_time)

    # save parameter names
    json.dump(
        model.parameters(),
        open(os.path.join(output_dir, 'params.json'), 'w')
    )  


if __name__ == '__main__':

    # IMPORTANT
    # You must first:
    # - Install multinest C libraries as per https://johannesbuchner.github.io/PyMultiNest/install.html#installing-pymultinest-and-pycuba
    # - Add LD_LIBRARY_PATH to environment (or use .profile)
    # export LD_LIBRARY_PATH=$HOME/repos/MultiNest/lib/:$LD_LIBRARY_PATH

    model_loc = 'data/interp_starforming_lambda_e_z.pickle'
    output_dir = 'out'
    run(model_loc, output_dir, resume=True)

    # To make corner plot: python ../PyMultiNest/multinest_marginals_corner.py out/
