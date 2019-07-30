import numpy as np
import matplotlib.pyplot as plt

def get_bounds(obs, wspec=None, initial_spec=None):
    photometry_wavelengths = obs["phot_wave"]
    xmin, xmax = np.min(photometry_wavelengths)*0.8, np.max(photometry_wavelengths)/0.8
    if wspec is not None:  # interpolate sed to calculate y bounds
        assert initial_spec is not None
        # evaluate wspec (x) vs. initial spec (y), along new x grid
        temp = np.interp(np.linspace(xmin,xmax,10000), wspec, initial_spec)
        ymin, ymax = temp.min()*0.8, temp.max()/0.4
    else:
        ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4
    return (xmin, xmax), (ymin, ymax)


def plot_obs_photometry(obs):
    plt.errorbar(obs["phot_wave"], obs['maggies'], yerr=obs['maggies_unc'], # observations, in observer frame
        label='Observed photometry',
        marker='o', markersize=10, alpha=0.8, ls='', lw=3,
        ecolor='red', markerfacecolor='none', markeredgecolor='red', 
        markeredgewidth=3)


def visualise_obs(obs):
    (xmin, xmax), (ymin, ymax) = get_bounds(obs)
    plt.figure(figsize=(16,8))

    plot_obs_photometry(obs)

    # plot_filters(ax, obs, ymin, ymax)

    # prettify
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()

def get_title(model):
    return ','.join(["{}={}".format(p, model.params[p][0]) 
                    for p in model.free_params])

def get_observer_frame_wavelengths(model, sps):
    # initial_phot is y-values (maggies) as observed at obs['phot_wave'] wavelengths, in observer frame?
    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting w_new = w_old * (1+z)
    # redshift the *restframe* sps spectral wavelengths
    source_wavelengths = sps.wavelengths  # wavelengths of (source frame) fluxes
    observer_wavelengths = source_wavelengths * a # redshift them via w_observed = w_source * (1+z), using z of model
    # wspec is now *observer frame* wavelengths of source fluxes
    return observer_wavelengths


def visualise_obs_and_model(obs, model, theta, sps): # apply theta to model/sps, not model.theta
    fig, ax = plt.subplots(figsize=(16,8))
    plot_obs_photometry(obs)
    plot_model_at_obs(ax, model, theta, obs, sps)
    plt.title(get_title(model))
    prettify(obs, fig, ax)


def visualise_obs_and_models(obs, model, theta_array, sps): # apply theta to model/sps, not model.theta
    fig, ax = plt.subplots(figsize=(16,8))
    assert len(theta_array.shape) == 2
    plot_obs_photometry(obs)
    for theta_row in range(theta_array.shape[0]):
        plot_model_at_obs(ax, model, theta_array[theta_row], obs, sps, trace=True)
    # plt.title(get_title(model))
    prettify(obs, fig, ax)


def prettify(obs, fig, ax):
    (xmin, xmax), (ymin, ymax) = get_bounds(obs)
    # (xmin, xmax), (ymin, ymax) = get_bounds(obs, wspec, initial_spec)
    plot_filters(ax, obs, ymin, ymax)
    # prettify
    ax.set_xlabel('Wavelength [A] (Observer Frame)')
    ax.set_ylabel('Flux Density [maggies]')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    fig.legend(loc='best', fontsize=20)
    fig.tight_layout()


def plot_model_at_obs(ax, model, theta, obs, sps, trace=False):
    model_spectrum, model_photometry, _ = model.sed(theta, obs=obs, sps=sps)  # call sps.get_spectrum()
    observer_spectral_wavelengths = get_observer_frame_wavelengths(model, sps)

    if trace:
        spectra_kwargs = dict(lw=0.3, color='k', alpha=0.05)
        photo_kwargs = dict(marker='o', alpha=0.01)
    else:
        spectra_kwargs = dict(label='Model spectrum', lw=0.7, color='navy', alpha=0.7)
        photo_kwargs = dict(label='Model photometry', marker='s', alpha=0.8)

    ax.loglog(observer_spectral_wavelengths, model_spectrum, **spectra_kwargs) # model spectra, observer frame
    ax.scatter(obs["phot_wave"], model_photometry, s=15., color='blue', **photo_kwargs) # model photometry, observer frame


def plot_filters(ax, obs, ymin, ymax):
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        ax.loglog(w, t, lw=3, color='gray', alpha=0.7)
