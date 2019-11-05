import numpy as np
from scipy.stats import norm
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt


def run_mcmc(lnprobfn, prior_center, prior_var, n_walkers=2, n_samples=5000):
    n_burnin = int(n_samples * 0.4)
    init_theta = prior_center + prior_var * np.random.rand(n_walkers, 1) - 0.5 # [-0.5, 0.5]

    # with Pool() as pool:
    pool = None
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, dim=1, lnpostfn=lnprobfn, pool=pool)
    sampler.run_mcmc(init_theta, n_samples)
    print(sampler.chain.shape)
    samples = sampler.chain[:, n_burnin:, :].flatten()
    print(samples.shape)
    return samples


def posterior_predictive(theta_samples, true_y, forward_model, deviation_metric, show=True):
    y_rep = forward_model(theta_samples)
    y_deviations = deviation_metric(true_y, theta_samples, forward_model)
    y_rep_deviations = deviation_metric(y_rep, theta_samples, forward_model)

    p_b = np.sum(y_rep_deviations > y_deviations)/len(y_deviations)
    print('P_b: {:.4f}'.format(p_b))

    alpha = 0.6
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
    _, bins, _ = ax0.hist(y_deviations, density=True, bins=30, alpha=alpha, label='y_deviations')
    ax0.hist(y_rep_deviations, density=True, bins=bins, alpha=alpha, label='y_rep_deviations')
    ax0.set_ylabel(r'$p($Deviation$|y$)')
    ax0.set_xlabel('Deviation')
    ax0.legend()

    ax1.hist2d(y_deviations, y_rep_deviations, bins=[40, 40])
    ax1.set_xlim([0., 5.])
    ax1.set_ylim([0., 5.])
    ax1.set_xlabel(r'$y$ deviations')
    ax1.set_ylabel(r'$y^{rep}$ deviations')

    fig.tight_layout()
    plt.show()


def deviation_metric(y, theta, forward_model):  # i.e. D(y, theta) or D(y^{rep}, theta)
    return np.abs(y - forward_model(theta))  # absolute difference between y and f(theta)


def guassian_test_case(true_observation):  # could be an object
    prior_var = 1.
    prior_center = 0.
    measurement_var = 1.

    def prior(mu):
        return norm(loc=prior_center, scale=prior_var).logpdf(mu)

    def forward_model(mu):
        # generate an observation from theta (with some fixed noise)
        # do not use in loglikelihood! Don't want to add noise there.
        return mu + np.random.normal(scale=measurement_var, size=mu.shape) 

    def loglikelihood(mu):
        return norm(mu, measurement_var).logpdf(true_observation)

    def mcmc_numerator(mu):
        return prior(mu) + loglikelihood(mu)  # in log space, hence '+'

    def conjugate_normal_posterior(observation):
        mean = prior_var / (measurement_var + prior_var) * observation + (measurement_var / (measurement_var + prior_var)) * prior_center
        var = 1 / (1/prior_var + 1/measurement_var)
        print('Conjugate: N({}, {})'.format(mean, var))
        return norm(loc=mean, scale=var)

    test=False
    if test:
        mu_test = np.linspace(0., 1.)
        prior_test = 10 ** prior(mu_test)
        loglikelihood_test = 10 ** loglikelihood(mu_test)
        plt.plot(mu_test, prior_test, label='prior')
        plt.plot(mu_test, loglikelihood_test, label='likelihood')
        plt.legend()
        plt.show()

    return prior, forward_model, loglikelihood, mcmc_numerator, conjugate_normal_posterior, prior_center, prior_var
    

def visualise_samples_vs_truth(samples, conjugate_normal_posterior):
    plt.hist(samples.flatten(), bins=40, density=True)
    x = np.linspace(samples.min(), samples.max(), 500)
    pdf = conjugate_normal_posterior.pdf(x)
    plt.plot(x, pdf, label='Analytic')
    plt.show()


def test_with_gaussian():
    true_y = 0.9
    _, forward_model, _, mcmc_numerator, conjugate_normal_posterior, prior_center, prior_var = guassian_test_case(true_y)
    samples = run_mcmc(mcmc_numerator, prior_center, prior_var)  # only for this test (honestly, I could remove)
    # visualise_samples_vs_truth(samples, conjugate_normal_posterior)
    posterior_predictive(samples, true_y, forward_model, deviation_metric, show=True)  # the important bit


if __name__ == '__main__':
    test_with_gaussian()
