import pickle

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

INTERPOLATED_QUASAR_LOC = 'data/quasar_template_interpolated.pickle'
INTERPOLATED_TORUS_LOC = 'data/torus_template_interpolated.pickle'

def load_quasar_template():
    # radio-quiet mean quasar template from
    # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz
    # TODO include sigma (std dev in observed sources used to construct this template) in lnprobfn
    df = pd.read_csv('data/quasar_template_shang.txt', skiprows=19, delim_whitespace=True)
    print('range: ', df['log_freq'].min(), df['log_freq'].max())
    log_freq_interp = interp1d(df['log_freq'], df['log_flux'], kind='linear')  # freq in log hz
    return log_freq_interp

def load_torus_template():
    df = pd.read_csv('data/selected_torus_template.csv')
    df['normalised_flux'] = df['flux'] * 30
    interp = interp1d(np.log10(df['wavelength']), np.log10(df['normalised_flux']), kind='linear')  # freq in log hz
    return interp

def load_interpolated_quasar_template():
    with open(INTERPOLATED_QUASAR_LOC, 'rb') as f:
        return pickle.load(f)

def load_interpolated_torus_template():
    with open(INTERPOLATED_TORUS_LOC, 'rb') as f:
        return pickle.load(f)

def eval_quasar_template(wavelengths, log_freq_interp, short_only=False):
    wavelengths_m = 1e-10 * wavelengths # from angstroms to m
    log_freqs = np.log10(299792458. / wavelengths_m)
    log_fluxes = log_freq_interp(log_freqs)
    fluxes = 10 ** log_fluxes
    if short_only:  # add exponential damping after 1 micron
        fluxes *= get_damping_multiplier(wavelengths, 'long')
    return fluxes


# TODO combine these functions
def eval_torus_template(wavelengths, interp, long_only=False):
    fluxes = 10 ** interp(np.log10(wavelengths))  # in angstroms still!
    if long_only:  # add exponential damping after 1 micron
        fluxes *= get_damping_multiplier(wavelengths, 'short')
    return fluxes


def get_damping_multiplier(wavelengths, damp):
    damping_multiplier = np.ones_like(wavelengths)
    if damp == 'long':  # damp wavelengths above 1 micron
        to_damp = wavelengths > 1e4
        log_m = -2
    elif damp == 'short':
        to_damp = wavelengths < 1e4
        log_m = 2
    else:
        raise ValueError('damp={} not understood'.format(damp))
    intercept = 1e4 ** (-1 * log_m)
    damping_multiplier[to_damp] = intercept * wavelengths[to_damp] ** log_m
    return damping_multiplier


if __name__ == '__main__':
    
    quasar_interp = load_quasar_template()
    with open(INTERPOLATED_QUASAR_LOC, 'wb') as f:
        pickle.dump(quasar_interp, f)
    del quasar_interp

    torus_interp = load_torus_template()
    with open(INTERPOLATED_TORUS_LOC, 'wb') as f:
        pickle.dump(torus_interp, f)
    del torus_interp

    quasar_interp = load_interpolated_quasar_template()
    torus_interp = load_interpolated_torus_template()

    eval_wavelengths = np.logspace(np.log10(1e2), np.log10(1e7), 500000) # in angstroms

    plt.loglog(eval_wavelengths, get_damping_multiplier(eval_wavelengths, 'long'), label='long')
    plt.loglog(eval_wavelengths, get_damping_multiplier(eval_wavelengths, 'short'), label='short')
    plt.legend()
    plt.savefig('results/damping_multiplier.png')
    plt.clf()

    plt.loglog(eval_wavelengths, eval_quasar_template(eval_wavelengths, quasar_interp), label='Original')
    plt.loglog(eval_wavelengths, eval_quasar_template(eval_wavelengths, quasar_interp, short_only=True), label='Without dust')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/quasar_template.png')
    # TODO need to fix the normalisation to be total flux under the curve, over all wavelengths
    plt.clf()

    plt.loglog(eval_wavelengths, eval_torus_template(eval_wavelengths, torus_interp), label='Original')
    plt.loglog(eval_wavelengths, eval_torus_template(eval_wavelengths, torus_interp, long_only=True), label='Without blue')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/torus_template.png')
    plt.clf()
    # TODO need to fix the normalisation to be total flux under the curve, over all wavelengths
    
    quasar_only = eval_quasar_template(eval_wavelengths, quasar_interp, short_only=True)
    torus_only = eval_torus_template(eval_wavelengths, torus_interp, long_only=True)
    net = quasar_only + torus_only
    plt.loglog(eval_wavelengths, quasar_only, label='Quasar Only')
    plt.loglog(eval_wavelengths, torus_only, label='Torus Only')
    plt.loglog(eval_wavelengths, net, label='Net')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/joint_quasar_torus_template.png')
    plt.clf()
