import logging

import numpy as np
import pandas as pd

from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis, SSPBasis

from agnfinder import quasar_templates, agn_models, extinction_models
from agnfinder.prospector import load_photometry


def build_cpz_obs(galaxy, **extras):
    """Build a dictionary of photometry (and eventually spectra?)
    Arguments are hyperparameters that are likely to change over runs

    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    obs = {}
    obs["filters"], obs["maggies"], obs['maggies_unc'] = load_photometry.load_maggies_from_galaxy(galaxy)
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

    # TODO increase galaxy extinction to 0.6

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]  # add sfh and tau
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
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    if extras['agn_mass']:  # True if either =True or =float, False if =None or =False
        logging.warning('Building custom CSPSpecBasisAGN as agn_mass is {}'.format(extras['agn_mass']))
        sps = CSPSpecBasisAGN(zcontinuous=zcontinuous, agn_mass=extras['agn_mass'], agn_eb_v=extras['agn_eb_v'], agn_torus_mass=extras['agn_torus_mass'])
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

    def __init__(self, *args, **kwargs):
        super().__init__()

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

        # call the original get_galaxy_spectrum method
        wave, spectrum, mass_frac = super().get_galaxy_spectrum(**params)

        # insert blue AGN template here into spectrum
        template_quasar_flux = self.quasar_template(wave, short_only=True)
        quasar_flux = template_quasar_flux * self.params['agn_mass'] * 1e14

        template_torus_flux = self.torus_template(wave, long_only=True)
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
        self.galaxy_flux = spectrum

        return wave, self.galaxy_flux + self.quasar_flux, mass_frac

    
    # def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        # """Get a spectrum and SED for the given params.
        # """

        # mass_weighted_smspec, mass_weighted_phot, mfrac = super().get_spectrum(outwave=None, filters=None, peraa=False, **params)

        # reverse the mass-weighting of agn component
