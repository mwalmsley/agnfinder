import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

Components = namedtuple('Components', ['wavelengths', 'galaxy', 'unextincted_quasar', 'extincted_quasar', 'torus', 'net'])

COMPONENT_COLORS = {
    'galaxy': 'g',
    'unextincted_quasar': 'b',
    'extincted_quasar': 'b',
    'torus': 'orange',
    'net': 'k'
}

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

def calculate_sed(model, theta, obs, sps):
    model_spectra, model_photometry, _ = model.sed(theta, obs=obs, sps=sps)  # call sps.get_spectrum()
    observer_wavelengths = get_observer_frame_wavelengths(model, sps)
    return model_spectra, model_photometry, observer_wavelengths


def plot_model_at_obs(ax, model, theta, obs, sps, trace=False):

    model_spectra, model_photometry, observer_wavelengths = calculate_sed(model, theta, obs, sps)

    if trace:
        spectra_kwargs = dict(lw=0.3, color='k', alpha=0.05)
        photo_kwargs = dict(marker='o', alpha=0.01)
    else:
        spectra_kwargs = dict(label='Model spectrum', lw=0.7, color='navy', alpha=0.7)
        photo_kwargs = dict(label='Model photometry', marker='s', alpha=0.8)

    ax.loglog(observer_wavelengths, model_spectra, **spectra_kwargs) # model spectra, observer frame
    ax.scatter(obs["phot_wave"], model_photometry, s=15., color='blue', **photo_kwargs) # model photometry, observer frame

def get_components(sps):
    return Components(
        wavelengths=sps.wavelengths,
        galaxy=sps.galaxy_flux,
        unextincted_quasar=sps.unextincted_quasar_flux,
        extincted_quasar=sps.extincted_quasar_flux,
        torus=sps.torus_flux,
        net=(sps.quasar_flux + sps.galaxy_flux)
    )


def calculate_many_components(model, theta_array, obs, sps, ax=None):
    if ax == None:
        _, ax = plt.subplots(figsize=(16, 6))
    all_components = []
    for theta in theta_array:
        _ = calculate_sed(model, theta, obs, sps)  # don't actually care, just triggering calculation
        all_components.append(get_components(sps))
    for component_name in all_components[0]._fields:
        if component_name not in {'wavelengths', 'unextincted_quasar'}:
            upper, lower = component_to_band([getattr(c, component_name) for c in all_components])
            ax.loglog(all_components[0].wavelengths, upper, color=COMPONENT_COLORS[component_name], alpha=0.3)
            ax.loglog(all_components[0].wavelengths, lower, color=COMPONENT_COLORS[component_name], alpha=0.3)
            ax.fill_between(all_components[0].wavelengths, lower, upper, color=COMPONENT_COLORS[component_name], alpha=0.3, label=component_name)
    ax.set_xlabel('Wavelength (A), Source Frame')
    ax.set_ylabel('Flux Density (before Dimming)')

def component_to_band(component_list):
    # assumes all wavelengths are the same
    component_array = np.array(component_list)
    upper_limit = np.percentile(component_array, .9, axis=0)
    lower_limit = np.percentile(component_array, .1, axis=0)
    return upper_limit, lower_limit
        


def plot_components(components, ax=None):
    if not ax:
        _, ax = plt.subplots(figsize=(16, 6))
    ax.loglog(components.wavelengths, components.galaxy, 'g', label='Galaxy')
    ax.loglog(components.wavelengths, components.unextincted_quasar, 'b--', label='Unextincted Quasar (not used)')
    ax.loglog(components.wavelengths, components.extincted_quasar, 'b', label='Extincted Quasar')
    ax.loglog(components.wavelengths, components.torus, 'orange', label='Torus')
    ax.loglog(components.wavelengths, components.net, 'k', label='Net (All)')
    ax.legend()
    ax.set_ylabel('Flux')
    ax.set_xlabel('Wavelength (A, restframe)')


def plot_filters(ax, obs, ymin, ymax):
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        ax.loglog(w, t, lw=3, color='gray', alpha=0.7)



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



def get_title(model):
    return ','.join(["{}={}".format(p, model.params[p][0]) 
                    for p in model.free_params])
