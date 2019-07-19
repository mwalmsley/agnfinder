import numpy as np
import matplotlib.pyplot as plt

def visualise_obs(obs):

    wphot = obs["phot_wave"]

    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    ymin, ymax = obs["maggies"].min()*0.8, obs["maggies"].max()/0.4
    plt.figure(figsize=(16,8))

    # plot all the data
    plt.plot(wphot, obs['maggies'],
         label='All observed photometry',
         marker='o', markersize=12, alpha=0.8, ls='', lw=3,
         color='slateblue')

    # overplot only the data we intend to fit
    mask = obs["phot_mask"]
    plt.errorbar(wphot[mask], obs['maggies'][mask], 
             yerr=obs['maggies_unc'][mask], 
             label='Photometry to fit',
             marker='o', markersize=8, alpha=0.8, ls='', lw=3,
             ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato', 
             markeredgewidth=3)

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


def visualise_obs_and_model(obs, model, theta, sps): # model.theta may not be theta_best?
    initial_spec, initial_phot, _ = model.sed(theta, obs=obs, sps=sps)  # do model.mean_model(theta) for sed, then observe?
    title_text = ','.join(["{}={}".format(p, model.params[p][0]) 
                       for p in model.free_params])

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    # photometric effective wavelengths
    wphot = obs["phot_wave"]
    # spectroscopic wavelengths
    if obs["wavelength"] is None:
        # *restframe* spectral wavelengths, since obs["wavelength"] is None
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]

    plt.figure(figsize=(16,8))
    # plot model + data
    plt.loglog(wspec, initial_spec, label='Model spectrum', 
        lw=0.7, color='navy', alpha=0.7)
    plt.errorbar(wphot, initial_phot, label='Model photometry', 
            marker='s',markersize=10, alpha=0.8, ls='', lw=3,
            markerfacecolor='none', markeredgecolor='blue', 
            markeredgewidth=3)
    plt.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'], 
            label='Observed photometry',
            marker='o', markersize=10, alpha=0.8, ls='', lw=3,
            ecolor='red', markerfacecolor='none', markeredgecolor='red', 
            markeredgewidth=3)
    plt.title(title_text)


    # establish bounds
    xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
    temp = np.interp(np.linspace(xmin,xmax,10000), wspec, initial_spec)
    ymin, ymax = temp.min()*0.8, temp.max()/0.4

    plot_filters(obs, ymin, ymax)

    # prettify
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux Density [maggies]')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend(loc='best', fontsize=20)
    plt.tight_layout()


def visualise_data_and_before_vs_after_minimisation():
    plt.figure(figsize=(16,8))

    # plot Data, best fit model, and old models
    plt.loglog(wspec, initial_spec, label='Old model spectrum',
        lw=0.7, color='gray', alpha=0.5)
    plt.errorbar(wphot, initial_phot, label='Old model Photometry', 
            marker='s', markersize=10, alpha=0.6, ls='', lw=3, 
            markerfacecolor='none', markeredgecolor='gray', 
            markeredgewidth=3)
    plt.loglog(wspec, pspec, label='Model spectrum', 
        lw=0.7, color='slateblue', alpha=0.7)
    plt.errorbar(wphot, pphot, label='Model photometry', 
            marker='s', markersize=10, alpha=0.8, ls='', lw=3,
            markerfacecolor='none', markeredgecolor='slateblue', 
            markeredgewidth=3)
    plt.errorbar(wphot, obs['maggies'], yerr=obs['maggies_unc'],
            label='Observed photometry', 
            marker='o', markersize=10, alpha=0.8, ls='', lw=3, 
            ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato', 
            markeredgewidth=3)

    plot_filters(obs, ymin, ymax)

    # Prettify
    plt.xlabel('Wavelength [A]')
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
