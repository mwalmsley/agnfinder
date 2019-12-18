import logging

import dill  # to pickle lambda functions
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt


class InterpolatedTemplate():

    def __init__(self, template_loc, data_loc=None):
        self.data_loc = data_loc
        self.template_loc = template_loc

        if self.data_loc is not None:
            logging.warning('Creating new template - will have side effect on disk!')
            self._interpolated_template = self._create_template()
            self._save_template()
        else:
            self._interpolated_template = self._load_template()


    def _create_template(self):
        raise NotImplementedError


    def _save_template(self):
        with open(self.template_loc, 'wb') as f:
            dill.dump(self._interpolated_template, f)


    def _load_template(self):
        with open(self.template_loc, 'rb') as f:
            return dill.load(f)


    def _eval_template(self):
        raise NotImplementedError


    def __call__(self, *args, **kwargs):
        return self._eval_template(*args, **kwargs)


class QuasarTemplate(InterpolatedTemplate):


    def _eval_template(self, wavelengths, short_only=False):
        fluxes = 10 ** self._interpolated_template(np.log10(wavelengths))
        if short_only:  # add exponential damping after 1 micron
            fluxes *= get_damping_multiplier(wavelengths, 'long')
        return fluxes

    def _create_template(self):
        # radio-quiet mean quasar template from
        # https://iopscience.iop.org/article/10.1088/0067-0049/196/1/2#apjs400220f6.tar.gz
        # TODO include sigma (std dev in observed sources used to construct this template) in lnprobfn
        df = pd.read_csv(self.data_loc, skiprows=19, delim_whitespace=True)
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


class TorusTemplate(InterpolatedTemplate):
    # TODO this will no longer be parameter-less (AGN mass scaling happening outside)
    # instead, should take inclination as an arg in eval_template?
    # bit messy with AGN mass externally though?

    def _eval_template(self, wavelengths, long_only=False):
        fluxes = 10 ** self._interpolated_template(np.log10(wavelengths))  # in angstroms still!
        if long_only:  # add exponential damping after 1 micron
            fluxes *= get_damping_multiplier(wavelengths, 'short')
        return fluxes


    def _create_template(self):
        df = pd.read_csv(self.data_loc)
        # enforce no flux below 100 angstroms
        df = df.append({'wavelength': 99.99, 'flux': 1e-15}, ignore_index=True)
        df = df.append({'wavelength': 1e-2, 'flux': 1e-15}, ignore_index=True)  
        # enforce no flux above 1e7 angstroms
        df = df.append({'wavelength': 10000000.1, 'flux': 1e-15}, ignore_index=True)
        df = df.append({'wavelength': 1e13, 'flux': 1e-15}, ignore_index=True) 
        # interpolate in log wavelength(A)/log freq(Hz) space
        df = df.sort_values('wavelength')
        # print(np.log10(df['wavelength']).min(), np.log10(df['wavelength']).max())
        interp = interp1d(
            np.log10(df['wavelength']),  # in angstroms
            np.log10(df['flux']),
            kind='linear'
        )  
        normalised_interp = normalise_template(interp)
        return normalised_interp


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
    # -21 so that agn mass is similar to galaxy mass (actually not the case? Not sure why -21, perhaps arbitrary)
    return lambda x: interp(x) - np.log10(total_flux) - 21

QUASAR_DATA_LOC = 'data/quasar_template_shang.txt'
TORUS_DATA_LOC = 'data/selected_torus_template.csv'

INTERPOLATED_QUASAR_LOC = 'data/quasar_template_interpolated.dill'
INTERPOLATED_TORUS_LOC = 'data/torus_template_interpolated.dill'

if __name__ == '__main__':


    # always create a fresh template (by providing data_loc)
    quasar = QuasarTemplate(INTERPOLATED_QUASAR_LOC, data_loc=QUASAR_DATA_LOC)
    torus = TorusTemplate(INTERPOLATED_TORUS_LOC, data_loc=TORUS_DATA_LOC)

    eval_wavelengths = np.logspace(np.log10(1e2), np.log10(1e8), 500000) # in angstroms

    plt.loglog(eval_wavelengths, get_damping_multiplier(eval_wavelengths, 'long'), label='long')
    plt.loglog(eval_wavelengths, get_damping_multiplier(eval_wavelengths, 'short'), label='short')
    plt.legend()
    plt.savefig('results/damping_multiplier.png')
    plt.clf()

    plt.loglog(eval_wavelengths, quasar(eval_wavelengths), label='Original')
    plt.loglog(eval_wavelengths, quasar(eval_wavelengths, short_only=True), label='Without dust')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/quasar_template.png')
    # TODO need to fix the normalisation to be total flux under the curve, over all wavelengths
    plt.clf()

    plt.loglog(eval_wavelengths, torus(eval_wavelengths), label='Original')
    plt.loglog(eval_wavelengths, torus(eval_wavelengths, long_only=True), label='Without blue')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux (normalised)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/torus_template.png')
    plt.clf()
    # TODO need to fix the normalisation to be total flux under the curve, over all wavelengths
    
    quasar_only = quasar(eval_wavelengths, short_only=True)
    torus_only = torus(eval_wavelengths, long_only=True)
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
