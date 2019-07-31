import dill  # to pickle lambda functions

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt

INTERPOLATED_QUASAR_LOC = 'data/quasar_template_interpolated.dill'
INTERPOLATED_TORUS_LOC = 'data/torus_template_interpolated.dill'

def load_quasar_template():
    # radio-quiet mean quasar template from
    # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz
    # TODO include sigma (std dev in observed sources used to construct this template) in lnprobfn
    df = pd.read_csv('data/quasar_template_shang.txt', skiprows=19, delim_whitespace=True)
    # print('log freq range: ', df['log_freq'].min(), df['log_freq'].max())
    freqs = 10 ** df['log_freq']
    wavelengths = 299792458. / freqs
    # print('log wavelength range: ', wavelengths.min(), wavelengths.max())
    # interpolate in log wavelength(A)/log freq(Hz) space
    interp = interp1d(  
        np.log10(wavelengths * 1e10),  # in angstroms
        df['log_flux'],
        kind='linear'
    )  
    normalised_interp = normalise_template(interp)
    return normalised_interp

def load_torus_template():
    df = pd.read_csv('data/selected_torus_template.csv')
    # enforce no flux below 100 angstroms
    df = df.append({'wavelength': 99.99, 'flux': 1e-15}, ignore_index=True)
    df = df.append({'wavelength': 1e-2, 'flux': 1e-15}, ignore_index=True)  
    # enforce no flux above 1e7 angstroms
    df = df.append({'wavelength': 10000000.1, 'flux': 1e-15}, ignore_index=True)
    df = df.append({'wavelength': 1e13, 'flux': 1e-15}, ignore_index=True) 
    # interpolate in log wavelength(A)/log freq(Hz) space
    df = df.sort_values('wavelength')
    print(np.log10(df['wavelength']).min(), np.log10(df['wavelength']).max())
    interp = interp1d(
        np.log10(df['wavelength']),  # in angstroms
        np.log10(df['flux']),
        kind='linear'
    )  
    normalised_interp = normalise_template(interp)
    return normalised_interp
    # return interp

def load_interpolated_quasar_template():
    with open(INTERPOLATED_QUASAR_LOC, 'rb') as f:
        return dill.load(f)

def load_interpolated_torus_template():
    with open(INTERPOLATED_TORUS_LOC, 'rb') as f:
        return dill.load(f)

def eval_quasar_template(wavelengths, interp, short_only=False):
    fluxes = 10 ** interp(np.log10(wavelengths))
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
        log_m = -5
    elif damp == 'short':
        to_damp = wavelengths < 1e4
        log_m = 5
    else:
        raise ValueError('damp={} not understood'.format(damp))
    intercept = 1e4 ** (-1 * log_m)
    damping_multiplier[to_damp] = intercept * wavelengths[to_damp] ** log_m
    return damping_multiplier


def normalise_template(interp):
    log_wavelengths = np.log10(np.logspace(np.log10(1e2), np.log10(1e7), 500000)) # in angstroms
    total_flux = simps(10 ** interp(log_wavelengths), 10 ** log_wavelengths, dx=1, even='avg')
    # return normalised flux in log space (remembering that division is subtraction)
    # -30 so that agn mass is similar to galaxy mass
    return lambda x: interp(x) - np.log10(total_flux) - 21


if __name__ == '__main__':
    
    quasar_interp = load_quasar_template()
    with open(INTERPOLATED_QUASAR_LOC, 'wb') as f:
        dill.dump(quasar_interp, f)
    del quasar_interp

    torus_interp = load_torus_template()
    with open(INTERPOLATED_TORUS_LOC, 'wb') as f:
        dill.dump(torus_interp, f)
    del torus_interp

    quasar_interp = load_interpolated_quasar_template()
    torus_interp = load_interpolated_torus_template()

    eval_wavelengths = np.logspace(np.log10(1e2), np.log10(1e8), 500000) # in angstroms

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
