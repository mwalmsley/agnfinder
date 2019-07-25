import logging

import numpy as np
import pandas as pd

from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.models.sedmodel import SedModel
from prospect.sources import CSPSpecBasis, SSPBasis

from agnfinder import quasar_template, agn_models
from agnfinder.prospector import load_photometry


def build_cpz_obs(galaxy, snr=10, **extras):
    """Build a dictionary of photometry (and eventually spectra?)
    Arguments are hyperparameters that are likely to change over runs
    
    :param snr:
        The S/N to assign to the photometry when calculating uncertainties (TODO replace)
        
    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    obs = {}
    obs["filters"], obs["maggies"], obs['maggies_unc'] = load_photometry.load_maggies_from_galaxy(galaxy, snr)
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


def build_model_demo_style(object_redshift=None, ldist=10.0, fixed_metallicity=None, add_duste=False, 
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
        
    :param add_duste: (optional, default: False)
        If `True`, add dust emission and associated (fixed) parameters to the model.
        
    :returns model:
        An instance of prospect.models.SedModel
    """

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]
    
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}
    
    # Let's make some changes to initial values appropriate for our objects and data
    model_params["zred"]["init"] = 0.0
    model_params["dust2"]["init"] = 0.05
    model_params["logzsol"]["init"] = -0.5
    model_params["tage"]["init"] = 13.
    model_params["mass"]["init"] = 1e8
    
    # These are dwarf galaxies, so lets also adjust the metallicity prior,
    # the tau parameter upward, and the mass prior downward
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    model_params["mass"]["disp_floor"] = 1e6
    model_params["tau"]["disp_floor"] = 1.0
    model_params["tage"]["disp_floor"] = 1.0
    
    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is not None:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity 

    if object_redshift is None:
        model_params["zred"]['isfree'] = True
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = 0.  # assume redshift 0 at first
        model_params["zred"]["prior"] = priors.TopHat(mini=0.0, maxi=0.3)
    else:
        # make sure zred is fixed
        model_params["zred"]['isfree'] = False
        # And set the value to the object_redshift keyword
        model_params["zred"]['init'] = object_redshift


    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, 
        # and `TemplateLibrary` returns dictionaries of parameter specifications, 
        # we can just update `model_params` with the parameters described in the 
        # pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])
        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model




def build_model(redshift=None, fixed_metallicity=None, dust=False, agn_fraction=None,
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

    # Get (a copy of) one of the prepackaged model set dictionaries.
    # This is, somewhat confusingly, a dictionary of dictionaries, keyed by parameter name
    model_params = TemplateLibrary["parametric_sfh"]
    
   # Now add the lumdist parameter by hand as another entry in the dictionary.
   # This will control the distance since we are setting the redshift to zero.  
   # In `build_obs` above we used a distance of 10Mpc to convert from absolute to apparent magnitudes, 
   # so we use that here too, since the `maggies` are appropriate for that distance.
    # model_params["lumdist"] = {"N": 1, "isfree": False, "init": ldist, "units":"Mpc"}
    
    # Let's make some changes to initial values appropriate for our objects and data
    # model_params["zred"]["init"] = 0.0
    # model_params["dust2"]["init"] = 0.05
    # model_params["logzsol"]["init"] = -0.5
    # model_params["tage"]["init"] = 13.
    # model_params["mass"]["init"] = 1e8
    
    # These are dwarf galaxies, so lets also adjust the metallicity prior,
    # the tau parameter upward, and the mass prior downward
    # model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    # model_params["tau"]["prior"] = priors.LogUniform(mini=1e-1, maxi=1e2)
    # model_params["mass"]["prior"] = priors.LogUniform(mini=1e6, maxi=1e10)

    # If we are going to be using emcee, it is useful to provide a 
    # minimum scale for the cloud of walkers (the default is 0.1)
    # model_params["mass"]["disp_floor"] = 1e6
    # model_params["tau"]["disp_floor"] = 1.0
    # model_params["tage"]["disp_floor"] = 1.0
    
    # Change the model parameter specifications based on some keyword arguments
    if fixed_metallicity is None:
        model_params['logzsol']['isfree'] = True
    else:
        # make it a fixed parameter
        model_params["logzsol"]["isfree"] = False
        #And use value supplied by fixed_metallicity keyword
        model_params["logzsol"]['init'] = fixed_metallicity 

    if redshift is None:
        # Set redshift as free
        model_params["zred"]['isfree'] = True
    else:
        # Set redshift as fixed to  keyword value
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = redshift

    if dust:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, 
        # and `TemplateLibrary` returns dictionaries of parameter specifications, 
        # we can just update `model_params` with the parameters described in the 
        # pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])

    if agn_fraction is None:
        logging.warning('No AGN fraction supplied - AGN not modelled')
    elif isinstance(agn_fraction, bool):
        if agn_fraction:
            logging.info('AGN fraction will be free parameter')  # TODO WARNING init set temporarily high!
            model_params['agn_fraction'] = {'N': 1, 'isfree': True, 'init': 0.9, 'prior': priors.TopHat(mini=0.0, maxi=1.0)}  # units?
        else:
            raise ValueError('agn_fraction set to False - use None instead!')
    elif isinstance(agn_fraction, float):
        logging.info('AGN fraction will be fixed at {}'.format(agn_fraction))
        model_params['agn_fraction'] = {'N': 1, 'isfree': False, 'init': agn_fraction}  # units?
    else:
        raise ValueError('agn fraction {} not understood'.format(agn_fraction))

        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model


def build_sps(zcontinuous=1, agn_fraction=None, **extras):
    """
    :param zcontinuous: 
        A vlue of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    if agn_fraction:  # True if either =True or =float, False if =None or =False
        logging.info('Building custom CSPSpecBasisAGN as agn_fraction is {}'.format(agn_fraction))
        sps = CSPSpecBasisAGN(zcontinuous=zcontinuous, agn_fraction=agn_fraction)
    else:
        logging.info('Building standard CSPSpec')
        sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps


class CSPSpecBasisAGN(CSPSpecBasis):
    """ 
    Override get_galaxy_spectrum to run as before but, before returning, add AGN component

    As with CSPSpecBasis, uses SSPBasis to implement get_spectrum(), which calls get_galaxy_spectrum and 
     - applies observational effects
     - normalises by mass
    """

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
        logging.debug('Using custom get_galaxy_spectrum with params {}'.format(params))
        self.update(**params)
        try:
            self.params['agn_fraction']
        except KeyError:
            raise AttributeError('Trying to calculate SED inc. AGN, but no `agn_fraction` parameter set')

        # call the original get_galaxy_spectrum method
        wave, spectrum, mass_frac = super(CSPSpecBasisAGN, self).get_galaxy_spectrum(**params)

        # insert AGN template here into spectrum
        interp = quasar_template.load_interpolated_quasar_template()
        normalised_quasar_flux = quasar_template.eval_quasar_template(wave, interp)
        quasar_flux = agn_models.scale_quasar_to_agn_fraction(initial_quasar_flux=normalised_quasar_flux, galaxy_flux=spectrum, agn_fraction=self.params['agn_fraction'])

        # TODO 

        return wave, spectrum + quasar_flux, mass_frac
