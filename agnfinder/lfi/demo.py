import numpy as np
import tensorflow as tf
import h5py
import corner
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.priors as priors
from agnfinder.prospector.main import save_samples

tf.logging.set_verbosity(tf.logging.ERROR)

def simulator(theta, seed=1, simulator_args=None, batch=1):
    return np.array([np.mean(theta) + 0.1 * np.random.rand()])

compressor_args=None
def compressor(d, compressor_args):
    # pydelfi can compress observables, but
    # don't compress because it's only 2d anyway
    return d


if __name__ == '__main__':

    simulator_args = None

    # example_theta = [0.30178471, 0.23758648, 0.32793839, 0.2836491,  0.23384894]
    # print(simulator(example_theta, seed=1, simulator_args=None, batch=None))
    # exit()

    # number of parameters
    n_inputs = 2
    # number of observables
    n_outputs = 1

    # specify the experimental observations
    observed_data = [.3]

    # we map all of our input parameters from [0,1] to the appropriate ranges
    lower = np.zeros(n_inputs)
    upper = np.ones(n_inputs)
    # we want to start with uniform priors
    prior = priors.Uniform(lower, upper)

    n_hiddens = [30, 30] #  hidden layers in MAF (must all be the same)
    # Build list of neural networks
    NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=n_inputs, n_data=n_outputs, n_hiddens=n_hiddens, n_mades=2, act_fun=tf.tanh, index=1)]


    DelfiEnsemble = delfi.Delfi(observed_data, prior, NDEs,
                                    param_limits=[lower, upper],
                                    param_names=['M1', 'TB', 'Mu', 'M2', 'Msl'],
                                    results_dir="data/lfi/demo_results",
                                    progress_bar=True,
                                   )

    print("DelfiEnsemble is ready! Begin training")

    # total number of points we want to evaluate
    total_sims = 500

    # number of points to evaluate with uniform priors (before training)
    n_initial = 125 

    # number of additional training runs
    n_populations = 3

    # size of each batch
    n_batch = int((total_sims-n_initial)/n_populations) 
    # with this, we have 3 runs with 100 points in each

    # sometimes (many times) chosen parameters are unphysical or too far
    # outside of our area of interest. In this case, the model returns [-inf, -inf]
    # Therefore, the model must have some backup points to evaluate at instead
    # safety is the number of backup points
    safety = int(9999/n_batch)  # must be less than 10k

    # number of times each point in the batch is used to train the neural networks
    n_epochs = 15 # 15 is small, usually a few hundred is good

    DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations,
                                  patience=max(1, int(3*n_epochs/20)),  # patience during each NN training sesh
                                  epochs=n_epochs,
                                  safety=safety,
                                  save_intermediate_posteriors=False)

    posterior_samples = DelfiEnsemble.emcee_sample()
    save_samples(posterior_samples, 'data/lfi/samples.hdf5')
    figure = corner.corner(posterior_samples)
    # , labels=['a', 'b'],
                        # show_titles=True, title_kwargs={"fontsize": 12})
    figure.savefig('data/lfi/corner.png')
