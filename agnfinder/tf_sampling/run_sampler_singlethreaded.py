import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf  # just for eager toggle

from agnfinder.prospector import load_photometry
from agnfinder.tf_sampling import run_sampler, deep_emulator


# TODO will change to some kind of unique id for each galaxy, rather than the index
def get_galaxies_without_results(n_galaxies):
    without_results = []
    i = 0
    while len(without_results) < n_galaxies:
        if not os.path.isfile(run_sampler.get_galaxy_save_file(i, save_dir)):
            without_results.append(i)
        i += 1
    return without_results


def record_performance_on_galaxies(checkpoint_loc, selected_catalog_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir):
    emulator = deep_emulator.get_trained_keras_emulator(deep_emulator.tf_model(), checkpoint_loc, new=False)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if selected_catalog_loc is not '':
        # real galaxies, selected from uK_IR sample by highest (?) random forest prob. (`Pr[{class}]_case_III')
        assert os.path.isfile(selected_catalog_loc)  # TODO generalise?
        df = pd.read_parquet(selected_catalog_loc)
        logging.info(f'Loading {len(df)} galaxies')
        rf_classes = ['passive', 'starforming', 'starburst', 'agn', 'qso', 'outlier']
        for c in rf_classes:
            logging.info('{}: {:.2f}'.format(c, df[f'Pr[{c}]_case_III'].sum()))

        true_observation = np.zeros((len(df), 12)).astype(np.float32)  # fixed bands
        redshifts = np.zeros((len(df), 1), dtype=np.float32)
        for n in tqdm(range(len(df))):
            galaxy = df.iloc[n]  # I don't know why iterrows is apparently returning one more row than len(df)??
            # TODO uncertainties are not yet used! See log prob, currently 5% uncertainty by default
            _, maggies, maggies_unc = load_photometry.load_maggies_from_galaxy(galaxy, reliable=True)
            true_observation[n] = maggies.astype(np.float32)
            redshifts[n] = galaxy['redshift']
        true_params = np.zeros((len(df), 7)).astype(np.float32)
        # true_params = None TODO quite awkward as I often use it in asserts or for expected param dim
        logging.warning(f'Using {len(df)} real galaxies - forcing n_chains from {n_chains} to {len(df)} accordingly')
        n_chains = len(df)  # overriding whatever the arg was
        galaxy_indices = df.index  # I should *really* reset the index beforehand so this is 1....33

    else:
        # fake galaxies, drawn from our priors and used as emulator training data
        logging.info('Using fake galaxies, drawn randomly from the hypercube')
        _, _, x_test, y_test = deep_emulator.data(cube_dir='data/cubes/latest')  # TODO could make as arg
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        # galaxy_indices = get_galaxies_without_results(n_chains)  # commenting out for now
        galaxy_indices = np.arange(n_chains)
        true_params = x_test[galaxy_indices, 1:]  # excluding the 0th redshift param, which we treat as fixed
        redshifts = x_test[galaxy_indices, :1].astype(np.float32)  # shape (n_galaxies, 1)
        true_observation = deep_emulator.denormalise_photometry(y_test[galaxy_indices]) 

    # print(redshifts.shape, true_observation.shape, true_params.shape)
    # print(true_observation)
    assert len(redshifts) == len(true_observation) == len(true_params)
    print(redshifts)
    print(redshifts[0])
    print(true_params[0])
    print(true_observation[0])
    run_sampler.sample_galaxy_batch(galaxy_indices, true_observation, redshifts, true_params, emulator, n_burnin, n_samples, n_chains, init_method, save_dir)


if __name__ == '__main__':

    """
    Run the emulated HMC method on many galaxies, in a single thread.
    Evaluating performance at recovering posteriors can be done in `evaluate_performance.py`

    Example use: 
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling --n-chains 4 --n-samples 100 --n-burnin 100 --init random
    python agnfinder/tf_sampling/run_sampler_singlethreaded.py --checkpoint-loc results/checkpoints/latest --output-dir results/emulated_sampling --selected data/uk_ir_selection_577.parquet

    """
    parser = argparse.ArgumentParser(description='Run emulated HMC on many galaxies')
    parser.add_argument('--checkpoint-loc', type=str, dest='checkpoint_loc')
    parser.add_argument('--output-dir', dest='output_dir', type=str)  # in which save_dir will be created
    parser.add_argument('--max-galaxies', type=int, default=1, dest='max_galaxies')
    parser.add_argument('--selected', type=str, default='', dest='selected_catalog_loc')
    parser.add_argument('--n-burnin', type=int, default=1000, dest='n_burnin')  # below 1000, may not find good step size
    parser.add_argument('--n-samples', type=int, default=6000, dest='n_samples')  # 6000 works well?
    parser.add_argument('--n-chains', type=int, default=96, dest='n_chains')  # 96 is ideal on my laptop, more memory = more chains free
    parser.add_argument('--init', type=str, dest='init_method', default='optimised', help='Can be one of: random, roughly_correct, optimised')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(logging.INFO)  # some third party library is mistakenly setting the logging somewhere...

    checkpoint_loc =  args.checkpoint_loc
    output_dir = args.output_dir
    assert checkpoint_loc is not None
    assert output_dir is not None
    max_galaxies = args.max_galaxies
    n_burnin = args.n_burnin
    n_samples = args.n_samples
    n_chains = args.n_chains
    init_method = args.init_method
    selected_catalog_loc = args.selected_catalog_loc

    if selected_catalog_loc is not '':
        save_dir = os.path.join(output_dir, os.path.basename(selected_catalog_loc))
    else:
        save_dir = os.path.join(output_dir, 'latest_{}_{}_{}'.format(n_samples, n_chains, init_method))

    record_performance_on_galaxies(checkpoint_loc, selected_catalog_loc, max_galaxies, n_burnin, n_samples, n_chains, init_method, save_dir)
    # run_sampler.aggregate_performance(save_dir, n_samples, chains_per_galaxy=1)
    # samples, true_params, true_observations = run_sampler.read_performance(save_dir)
    # print(samples.shape, true_params.shape, true_observations.shape)
