import numpy as np
import matplotlib.pyplot as plt
from agnfinder.quasar_template import load_quasar_template, eval_quasar_template

from agnfinder.prospector import cpz_builders


def make_galaxy_sed():

    params = {
        'zred': 0.07033542490061963,
        'mass': 14361224363.648266,
        'logzsol': -1.5000000366704935,
        'dust2': 0.38205777251182615,
        'tage': 9.017467356520612,
        'tau': 1.9203743725619529
    }

    sps = cpz_builders.SSPBasis(**params)

    flux, _, mfrac = sps.get_spectrum()
    
    np.savetxt('flux_maggies.npy', flux)

    wavelengths = sps.wavelengths
    np.savetxt('wavelengths.npy', wavelengths)

def load_galaxy_sed():
    return np.loadtxt('wavelengths.npy'), np.loadtxt('flux_maggies.npy')


def scale_quasar_to_agn_fraction(galaxy_flux, initial_quasar_flux, agn_fraction):
    # make each sed similar, and create a net SED at agn fraction %
    total_galaxy_flux = np.sum(galaxy_flux)
    total_quasar_flux = np.sum(initial_quasar_flux)
    target_quasar_flux = total_galaxy_flux * agn_fraction
    return initial_quasar_flux * target_quasar_flux / total_quasar_flux


if __name__ == '__main__':

    # make_galaxy_sed()

    galaxy_wavelength, galaxy_flux = load_galaxy_sed()
    mfrac = 0.5843067062572432  # manually, awkward to save a scalar

    plt.loglog(galaxy_wavelength, galaxy_flux, label='galaxy')
    interp = load_quasar_template()
    quasar_normalized_flux = eval_quasar_template(galaxy_wavelength, interp)
    plt.loglog(galaxy_wavelength, quasar_normalized_flux, label='quasar')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/galaxy_and_quasar_template.png')

    plt.clf()

    quasar_flux_scaled = scale_quasar_to_agn_fraction(galaxy_flux, quasar_normalized_flux, agn_fraction=0.5)
    plt.loglog(galaxy_wavelength, galaxy_flux, label='galaxy')
    plt.loglog(galaxy_wavelength, quasar_flux_scaled, label='quasar')
    plt.loglog(galaxy_wavelength, quasar_flux_scaled + galaxy_flux, label='net (50% AGN Frac.')
    plt.xlabel('Wavelength')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/galaxy_50%_agn.png')
