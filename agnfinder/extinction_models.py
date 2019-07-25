import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def smc_extinction(wavelength, flux, eb_v, k_l_interp):
    return flux * 10 ** (-0.4 * k_l_interp(wavelength) * eb_v)


INTERPOLATED_SMC_EXTINCTION_LOC = 'data/interpolated_smc_extinction.pickle'
def load_interpolated_smc_extinction():
    with open(INTERPOLATED_SMC_EXTINCTION_LOC, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    
    df = pd.read_csv('data/smc_extinction_prevot_1984.dat', delim_whitespace=True)
    print(df.head())
    print(df.columns.values)
    k_l_interp = interp1d(df['wavelength'], df['k_l'], kind='linear')

    with open(INTERPOLATED_SMC_EXTINCTION_LOC, 'wb') as f:
        pickle.dump(k_l_interp, f)
    del k_l_interp

    k_l_interp = load_interpolated_quasar_template()

    eb_v_values = list(np.linspace(0.1, 0.5, 5))
    eval_wavelengths = np.logspace(np.log10(1000), np.log10(40000), 5000) # in angstroms
    uniform_flux = np.ones_like(eval_wavelengths)
    # print(eval_wavelengths.min(), eval_wavelengths.max())
    plt.loglog(eval_wavelengths, uniform_flux, 'k--', label='Initial')
    for eb_v in eb_v_values:
        plt.loglog(eval_wavelengths, smc_extinction(eval_wavelengths, uniform_flux, eb_v, k_l_interp), label='Extincted (EB_V={:.1f})'.format(eb_v))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/smc_extinction.png')
