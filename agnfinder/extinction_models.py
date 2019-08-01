import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from agnfinder import quasar_templates

class ExtinctionTemplate(quasar_templates.InterpolatedTemplate):

    def _create_template(self):
        df = pd.read_csv(self.data_loc, delim_whitespace=True)
        # no extinction outside this range. Warning
        return interp1d(df['wavelength'], df['k_l'], kind='linear', fill_value=0, bounds_error=False)  


    def _eval_template(self, wavelength, flux, eb_v):
        return flux * 10 ** (-0.4 * self._interpolated_template(wavelength) * eb_v)


SMC_DATA_LOC = 'data/smc_extinction_prevot_1984.dat'
INTERPOLATED_SMC_EXTINCTION_LOC = 'data/interpolated_smc_extinction.dill'

if __name__ == '__main__':
    
    smc_extinction = ExtinctionTemplate(template_loc=INTERPOLATED_SMC_EXTINCTION_LOC, data_loc=SMC_DATA_LOC)

    eb_v_values = list(np.linspace(0.1, 0.5, 5))
    eval_wavelengths = np.logspace(np.log10(100), np.log10(40000), 5000) # in angstroms
    uniform_flux = np.ones_like(eval_wavelengths)

    plt.loglog(eval_wavelengths, uniform_flux, 'k--', label='Initial')
    for eb_v in eb_v_values:
        plt.loglog(eval_wavelengths, smc_extinction(eval_wavelengths, uniform_flux, eb_v), label='Extincted (EB_V={:.1f})'.format(eb_v))
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/smc_extinction.png')
