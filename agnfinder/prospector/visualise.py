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


def visualise_obs(obs):
    (xmin, xmax), (ymin, ymax) = get_bounds(obs)
    plt.figure(figsize=(16,8))
    plt.plot(obs["phot_wave"], obs['maggies'],
         label='All observed photometry',
         marker='o', markersize=12, alpha=0.8, ls='', lw=3,
         color='slateblue')

    plot_filters(obs, ymin, ymax)

    # prettify
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()


def visualise_obs_and_model(obs, model, theta, sps): # apply theta to model/sps, not model.theta

    # model 
    initial_spec, initial_phot, _ = model.sed(theta, obs=obs, sps=sps)  # call sps.get_spectrum()
    # initial_phot is y-values (maggies) as observed at obs['phot_wave'] wavelengths, in observer frame?
    title_text = ','.join(["{}={}".format(p, model.params[p][0]) 
                       for p in model.free_params])

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting w_new = w_old * (1+z)

    # observations, perhaps interpreted at model z

    # photometric effective wavelengths
    photometric_wavelengths = obs["phot_wave"]

    # spectroscopic wavelengths
    # observed_spectroscopic_wavelengths = obs["wavelength"]  # observer frame, by definition

    # if observed_spectroscopic_wavelengths is None:  # no spectroscopy (always true for me)
        # redshift the *restframe* sps spectral wavelengths
    wspec = sps.wavelengths  # wavelengths of (source frame) fluxes
    wspec *= a # redshift them via w_observed = w_source * (1+z), using z of model
    # wspec is now *observer frame* wavelengths of source fluxes

    # else:
    #     wspec = observed_spectroscopic_wavelengths

    plt.figure(figsize=(16,8))
    # plot model + data
    plt.loglog(wspec, initial_spec, label='Model spectrum', # model spectra, observer frame
        lw=0.7, color='navy', alpha=0.7)
    plt.errorbar(photometric_wavelengths, initial_phot, label='Model photometry', # model photometry, observer frame
            marker='s',markersize=10, alpha=0.8, ls='', lw=3,
            markerfacecolor='none', markeredgecolor='blue', 
            markeredgewidth=3)
    plt.errorbar(photometric_wavelengths, obs['maggies'], yerr=obs['maggies_unc'], # observations, in observer frame
            label='Observed photometry',
            marker='o', markersize=10, alpha=0.8, ls='', lw=3,
            ecolor='red', markerfacecolor='none', markeredgecolor='red', 
            markeredgewidth=3)
    plt.title(title_text)

    (xmin, xmax), (ymin, ymax) = get_bounds(obs, wspec, initial_spec)

    plot_filters(obs, ymin, ymax)

    # prettify
    plt.xlabel('Wavelength [A] (Observer Frame)')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()


def plot_filters(obs, ymin, ymax):
    for f in obs['filters']:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        plt.loglog(w, t, lw=3, color='gray', alpha=0.7)
