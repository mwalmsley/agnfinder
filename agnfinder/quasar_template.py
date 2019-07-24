import pickle

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

INTERPOLATED_QUASAR_LOC = 'data/quasar_template_interpolated.pickle'

def load_quasar_template():
    # radio-quiet mean quasar template from
    # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz
    # TODO include sigma (std dev in observed sources used to construct this template) in lnprobfn
    df = pd.read_csv('data/quasar_template_shang.txt', skiprows=19, delim_whitespace=True)
    print('range: ', df['log_freq'].min(), df['log_freq'].max())
    log_freq_interp = interp1d(df['log_freq'], df['log_flux'], kind='linear')  # freq in log hz
    return log_freq_interp

    # freq = 10 ** (df['log_freq'])  # now in Hz
    # # print('frequencies', freq.min(), freq.max())
    # wavelength = 299792458 / freq
    # # print('wavelengths', wavelength.min(), wavelength.max())
    # flux = 10 ** (df['log_flux'])  # mu Fmu arbitrary units (normalised)
    # interp = interp1d(wavelength, flux, kind='quadratic')
    # return interp  # can be evaluated at new wavelengths

def load_interpolated_quasar_template():
    with open(INTERPOLATED_QUASAR_LOC, 'rb') as f:
        return pickle.load(f)

def eval_quasar_template(wavelengths, log_freq_interp):
    wavelengths_m = 1e-10 * wavelengths # from angstroms to m
    log_freqs = np.log10(299792458. / wavelengths_m) 
    # print(log_freqs.min(), log_freqs.max())
    log_fluxes = log_freq_interp(log_freqs)
    return 10 ** log_fluxes

if __name__ == '__main__':
    
    interp = load_quasar_template()

    with open(INTERPOLATED_QUASAR_LOC, 'wb') as f:
        pickle.dump(interp, f)
    del interp

    interp = load_interpolated_quasar_template()

    eval_wavelengths_a = np.logspace(np.log10(1e2), np.log10(1e8), 500000) # in angstroms
    # print(eval_wavelengths.min(), eval_wavelengths.max())
    plt.loglog(eval_wavelengths_a, eval_quasar_template(eval_wavelengths_a, interp))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.tight_layout()
    plt.savefig('results/quasar_template.png')
    