import logging
import os

import numpy as np

import fsps
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis

from agnfinder import quasar_templates, extinction_models
from agnfinder.prospector import load_photometry
from agnfinder.fsps_emulation import emulate


def build_cpz_obs(galaxy, reliable, **extras):
    """Build a dictionary of photometry (and eventually spectra?)
    Arguments are hyperparameters that are likely to change over runs

    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    obs = {}
    obs["filters"], obs["maggies"], obs['maggies_unc'] = load_photometry.load_maggies_from_galaxy(galaxy, reliable)
    print(obs['filters'])
    # Now we need a mask, which says which flux values to consider in the likelihood.
    # IMPORTANT: the mask is *True* for values that you *want* to fit
    obs["phot_mask"] = np.array([True for _ in obs['filters']])

    # This is an array of effective wavelengths for each of the filters.  
    # It is not necessary, but it can be useful for plotting so we store it here as a convenience
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])

    # We do not have a spectrum, so we set some required elements of the obs dictionary to None.
    # (this would be a vector of vacuum wavelengths in angstroms)
    # Note: could use the SDSS spectra here for truth label fitting
    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None

    # This function ensures all required keys are present in the obs dictionary,
    # adding default values if necessary
    obs = fix_obs(obs)

    return obs


def build_model(redshift, fixed_metallicity=None, dust=False, agn_mass=None, agn_eb_v=None, agn_torus_mass=None, igm_absorbtion=True,
            **extras):
    """Build a prospect.models.SedModel object
    
    :param object_redshift: (optional, default: None)
        If given, produce spectra and observed frame photometry appropriate 
        for this redshift. Otherwise, the redshift will be zero.
        
    :param ldist: (optional, default: 10)
        The luminosity distance (in Mpc) for the model.  Spectra and observed 
        frame (apparent) photometry will be appropriate for this luminosity distance.
        
    :param fixed_metallicity: (optional, default: None)
        If given, fix the model metallicity (:math:`log(Z/Z_sun)`) to the given value.
        
    :param dust: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """
    logging.debug(redshift)
    logging.debug(fixed_metallicity)
    logging.debug(dust)
    logging.debug(agn_mass)
    logging.debug(agn_eb_v)
    logging.debug(agn_torus_mass)
    logging.debug(igm_absorbtion)
    logging.debug(extras)
    # TODO increase galaxy extinction to 0.6

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]  # add sfh and tau
    # dust2 init -.6, uniform [0, 2] uniform prior
    # delay-tau model with tau [0.1, 30] log-uniform prior
    # tage (burst start?) init 1, uniform (tophat) [0.001, 13.8]

    model_params['dust_type'] = {"N": 1, "isfree": False, "init": 2}  # Calzetti, as opposed to 0 for power law 
    # fixed: Kroupa IMF

    # mass: lower 10^9, upper 10^12
    # TODO set to logzsol=0 fixed, make this assumption 
    # dust (optical depth at 5500A): init=0.6, prior [0, 2], hopefully this is okay?


    # sfh: 'delay-tau' is 4
    # _parametric_["tau"]  = {"N": 1, "isfree": True,
    #                     "init": 1, "units": "Gyr^{-1}",
    #                     "prior": priors.LogUniform(mini=0.1, maxi=30)}


    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is None:
        model_params['logzsol']['isfree'] = True
    else:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity 
    
    if dust:
        # Add dust emission (with fixed dust SED parameters)
        logging.info('Including dust emission fixed parameters')
        model_params.update(TemplateLibrary["dust_emission"])
    if igm_absorbtion:
        # Add (fixed) IGM absorption parameters: madau attenuation (fixed to 1.)
        model_params.update(TemplateLibrary["igm"])


    # my convention: if None, don't model. if value, model as fixed to value. If True, model as free.

    if redshift is None:
        raise ValueError('Redshift set to None - must be included in model!')
    if isinstance(redshift, float):
        logging.info('Using fixed redshift of {}'.format(redshift))
        # Set redshift as fixed to value
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = redshift
    else:
        logging.info('Using free redshift')
        # Set redshift as free
        assert redshift == True # not just truthy, but exactly True/bool
        model_params["zred"]['isfree'] = True

    if agn_mass is None:
        logging.warning('No AGN mass supplied - AGN not modelled')
    elif isinstance(agn_mass, float):
        logging.info('AGN mass will be fixed at {}'.format(agn_mass))
        model_params['agn_mass'] = {'N': 1, 'isfree': False, 'init': agn_mass}  # units? 
    else:
        assert agn_mass == True
        logging.info('AGN mass will be free parameter')
        model_params['agn_mass'] = {'N': 1, 'isfree': True, 'init': 1., 'prior': priors.LogUniform(mini=1e-7, maxi=15)}
    if agn_eb_v is None:
        logging.warning('AGN extinction not modelled')
    else:
        assert agn_mass

        if isinstance(agn_eb_v, float):
            logging.info('Using fixed agn_eb_v of {}'.format(agn_eb_v))
            assert agn_mass
            model_params['agn_eb_v'] = {"N": 1, "isfree": False, "init": agn_eb_v, "units":"", 'prior': priors.TopHat(mini=0., maxi=0.5)}
        else:
            logging.info('Using free agn_eb_v')
            assert agn_mass == True
            model_params['agn_eb_v'] = {"N": 1, "isfree": True, "init": 0.1, "units":"", 'prior': priors.TopHat(mini=0., maxi=0.5)}
        
        if agn_torus_mass is None:
            logging.warning('Not modelling AGN torus')
        elif isinstance(agn_torus_mass, float):
            logging.info('Using fixed obscured torus of {}'.format(agn_torus_mass))
            model_params['agn_torus_mass'] = {"N": 1, "isfree": False, "init": agn_torus_mass, "units":"", 'prior': priors.LogUniform(mini=1e-7, maxi=15)}
        else:
            logging.info('Using free obscured torus')
            model_params['agn_torus_mass'] = {"N": 1, "isfree": True, "init": .1, "units":"", 'prior': priors.LogUniform(mini=1e-7, maxi=15)}
        

    # explicitly no FSPS dusty torus
    # model_params['fagn'] = None
    # model_params['agn_tau'] = None

    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model


def build_sps(zcontinuous=1, **extras):
    """
    :param zcontinuous: 
        A value of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    if extras['agn_mass']:  # True if either =True or =float, False if =None or =False
        logging.warning('Building custom CSPSpecBasisAGN as agn_mass is {}'.format(extras['agn_mass']))
        sps = CSPSpecBasisAGN(
            zcontinuous=zcontinuous,
            agn_mass=extras['agn_mass'], 
            agn_eb_v=extras['agn_eb_v'],
            agn_torus_mass=extras['agn_torus_mass'],
            emulate_ssp=extras['emulate_ssp']
        )
    else:
        logging.warning('Building standard CSPSpec')
        sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps


class CSPSpecBasisAGN(CSPSpecBasis):
    """ 
    Override get_galaxy_spectrum to run as before but, before returning, add AGN component

    As with CSPSpecBasis, uses SSPBasis to implement get_spectrum(), which calls get_galaxy_spectrum and 
     - applies observational effects
     - normalises by mass
    """

    # exactly as with super, one change
    def __init__(self, zcontinuous=1, reserved_params=['zred', 'sigma_smooth'],
                 vactoair_flag=False, compute_vega_mags=False, emulate_ssp=False, **kwargs):

        # super().__init__()
        
        if emulate_ssp:
            logging.warning('Using custom FSPS emulator for SSP')
            self.ssp = CustomSSP()
        else:
            logging.warning('Using standard FSPS for SSP, no emulation')
            self.ssp = fsps.StellarPopulation(compute_vega_mags=compute_vega_mags,
                                            zcontinuous=zcontinuous,
                                            vactoair_flag=vactoair_flag)

        self.reserved_params = reserved_params
        self.params = {}
        self.update(**kwargs)
        # custom init from here

        # TODO could wrap these globals entirely within quasar_templates
        self.quasar_template = quasar_templates.QuasarTemplate(template_loc=quasar_templates.INTERPOLATED_QUASAR_LOC)
        self.torus_template = quasar_templates.TorusTemplate(template_loc=quasar_templates.INTERPOLATED_TORUS_LOC)
        self.extinction_template = extinction_models.ExtinctionTemplate(template_loc=extinction_models.INTERPOLATED_SMC_EXTINCTION_LOC)

        self.galaxy_flux = None
        self.unextincted_quasar_flux = None
        self.quasar_flux = None
        self.torus_flux = None
        self.extincted_quasar_flux = None


    def get_galaxy_spectrum(self, **params):
        """Update parameters, then multiply SSP weights by SSP spectra and
        stellar masses, and sum.
        :returns wave:
            Wavelength in angstroms.
        :returns spectrum:
            Spectrum in units of Lsun/Hz/solar masses formed.
        :returns mass_fraction:
            Fraction of the formed stellar mass that still exists.
        """

        # don't log unless you need do, it's slow!
        # logging.debug('Using custom get_galaxy_spectrum with params {}'.format(params))
        
        self.update(**params)
        try:
            self.params['agn_mass']
        except KeyError:
            raise AttributeError('Trying to calculate SED inc. AGN, but no `agn_mass` parameter set')

        """Copy of get_galaxy_spectrum"""
        self.update(**params)  # store all arguments (model params) in self.params

        mass = np.atleast_1d(self.params['mass']).copy()
        mfrac = np.zeros_like(mass)
        self.update_component(0)
        wave, spectrum = self.ssp.get_spectrum(
            tage=self.ssp.params['tage'],
            peraa=False
        )
        mfrac_sum = self.ssp.stellar_mass 

        # rename to match
        mass_frac = mfrac_sum
        stellar_spectrum = spectrum

        # insert blue AGN template here into spectrum
        template_quasar_flux = self.quasar_template(wave, short_only=True)  # normalised scale
        quasar_flux = template_quasar_flux * self.params['agn_mass'] * 1e14

        template_torus_flux = self.torus_template(wave, long_only=True)  # normalised scale
        torus_flux = template_torus_flux * self.params['agn_torus_mass'] * 1e14

        # must always be specified, even if None
        if self.params['agn_eb_v']:  # float will eval as True
            extincted_quasar_flux = self.extinction_template(wave, quasar_flux, self.params['agn_eb_v'])
        else:  # don't model
            extincted_quasar_flux = quasar_flux

        self.unextincted_quasar_flux = quasar_flux
        self.extincted_quasar_flux = extincted_quasar_flux
        self.torus_flux = torus_flux
        self.quasar_flux = extincted_quasar_flux + torus_flux
        self.galaxy_flux = stellar_spectrum

        return wave, self.galaxy_flux + self.quasar_flux, mass_frac


    # def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        # """Get a spectrum and SED for the given params.
        # """

        # mass_weighted_smspec, mass_weighted_phot, mfrac = super().get_spectrum(outwave=None, filters=None, peraa=False, **params)

        # reverse the mass-weighting of agn component


# fsps.StellarPopulation
class CustomSSP():

    def __init__(self, careful=True):
        logging.warning('Using cached SSP!')
        # TODO may need to mock other properties e.g. params
        
        # TODO hardcoded paths for now
        model_dir = 'notebooks'
        num_params = 3
        num_bases = 10
        gp_model_loc = os.path.join(model_dir, 'gpfit_'+str(num_bases)+'_'+str(num_params) + '.zip')
        pca_model_loc = os.path.join(model_dir, 'pcaModel.pickle')
        self._spectrum_emulator = emulate.GPEmulator(
            gp_model_loc=gp_model_loc,
            pca_model_loc=pca_model_loc
        )
        mass_model_loc = os.path.join(model_dir, 'mass_emulator.pickle')
        self._mass_emulator = emulate.SKLearnEmulator(model_loc=mass_model_loc)
        reference_wave_loc = os.path.join(model_dir, 'reference_wave.txt')
        self.wavelengths = np.loadtxt(reference_wave_loc)  # use a fixed array
        self.stellar_mass = None  # already exists!
        self.params = CustomFSPSParams()  # mimicking how FSPS works, with args passed via dict (kind of a pain)
        self.careful = careful

    def get_spectrum(self, tage, peraa=False):
        if self.careful:
            assert tage is not 0  # not emulated!
            self.check_fixed_params_unchanged()
        param_vector = np.array([self.params['tau'], tage, self.params['dust2']])
        # emulator doesn't model the first 100 wavelengths (which are ~0) because of how Nesar made it, add them back manually
        spectra = np.hstack([np.ones(100) * 1e-60, self._spectrum_emulator(param_vector)])
        self.stellar_mass = self._mass_emulator(param_vector)  # mimicking FSPS also, so prospector can request self.stellar_mass
        return self.wavelengths, spectra

    def check_fixed_params_unchanged(self):
        expected_fixed_args = {
            'logzsol': 0.0, 
            'sfh': 4,
            'imf_type': 2, 
            'dust_type': 2, 
            'add_dust_emission': True, 
            'duste_umin': 1.0,
            'duste_qpah': 4.0, 
            'duste_gamma': 0.001, 
            'add_igm_absorption': True, 
            'igm_factor': 1.0 
        }
        for key, value in expected_fixed_args.items():
            assert self.params[key] == value


class CustomFSPSParams():

    def __init__(self):

        self.ssp_params = ["imf_type", "imf_upper_limit", "imf_lower_limit",
                    "imf1", "imf2", "imf3", "vdmc", "mdave",
                    "dell", "delt", "sbss", "fbhb", "pagb",
                    "add_stellar_remnants", "tpagb_norm_type",
                    "add_agb_dust_model", "agb_dust", "redgb", "agb",
                    "masscut", "fcstar", "evtype", "smooth_lsf"]

        self.csp_params = ["smooth_velocity", "redshift_colors",
                    "compute_light_ages","nebemlineinspec",
                    "dust_type", "add_dust_emission", "add_neb_emission",
                    "add_neb_continuum", "cloudy_dust", "add_igm_absorption",
                    "zmet", "sfh", "wgp1", "wgp2", "wgp3",
                    "tau", "const", "tage", "fburst", "tburst",
                    "dust1", "dust2", "logzsol", "zred", "pmetals",
                    "dust_clumps", "frac_nodust", "dust_index", "dust_tesc",
                    "frac_obrun", "uvb", "mwr", "dust1_index",
                    "sf_start", "sf_trunc", "sf_slope", "duste_gamma",
                    "duste_umin", "duste_qpah", "sigma_smooth",
                    "min_wave_smooth", "max_wave_smooth", "gas_logu",
                    "gas_logz", "igm_factor", "fagn", "agn_tau"]

        self._params = {}

    @property
    def all_params(self):
        return self.ssp_params + self.csp_params

    def __getitem__(self, k):
        return self._params[k]

    def __setitem__(self, k, v):
        self._params[k] = v
