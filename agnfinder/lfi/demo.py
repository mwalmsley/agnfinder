import numpy as np
import simulators.jla_supernovae.jla_simulator as jla
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import pydelfi.score as score
import pydelfi.priors as priors
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

def simulator(theta, seed, simulator_args, batch):
    return JLASimulator.simulation(theta, seed)

def compressor(d, compressor_args):
    return Compressor.scoreMLE(d)

if __name__ == '__main__':

    JLASimulator = jla.JLA_Model()

    simulator_args = None

    lower = np.array([0, -1.5, -20, 0, 0, -0.5])
    upper = np.array([0.6, 0, -18, 1, 6, 0.5])
    prior = priors.Uniform(lower, upper)

    theta_fiducial = np.array([0.2, -0.75, -19.05, 0.125, 2.65, -0.05])

    mu = JLASimulator.apparent_magnitude(theta_fiducial)
    Cinv = JLASimulator.Cinv

    h = np.array(abs(theta_fiducial))*0.01
    dmudt = JLASimulator.dmudt(theta_fiducial, h)

    Compressor = score.Gaussian(len(JLASimulator.data), theta_fiducial, mu = mu, Cinv = Cinv, dmudt = dmudt)
    Compressor.compute_fisher()
    Finv = Compressor.Finv


    compressor_args=None

    compressed_data = compressor(JLASimulator.data, compressor_args)

    NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=6, n_data=6, n_hiddens=[50,50], n_mades=5, act_fun=tf.tanh, index=5)]

    DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs, 
                                Finv = Finv, 
                                theta_fiducial = theta_fiducial, 
                                param_limits = [lower, upper],
                                param_names = ['\\Omega_m', 'w_0', 'M_\mathrm{B}', '\\alpha', '\\beta', '\\delta M'], 
                                results_dir = "simulators/jla_supernovae/results/",
                                input_normalization="fisher")

    DelfiEnsemble.fisher_pretraining()

    n_initial = 200
    n_batch = 200
    n_populations = 10

    DelfiEnsemble.sequential_training(simulator, compressor, n_initial, n_batch, n_populations, patience=20,
                        save_intermediate_posteriors=False)

    posterior_samples = DelfiEnsemble.emcee_sample()

    DelfiEnsemble.triangle_plot(samples=[posterior_samples])
