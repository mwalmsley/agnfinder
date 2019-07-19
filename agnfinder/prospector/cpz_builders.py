import sedpy
import numpy as np
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from prospect.models.sedmodel import SedModel

def build_cpz_obs(galaxy, snr=10, **extras):
    """Build a dictionary of photometry (and eventually spectra?)
    Arguments are hyperparameters that are likely to change over runs
    
    :param snr:
        The S/N to assign to the photometry when calculating uncertainties (TODO replace)
        
    :returns obs:
        A dictionary of observational data to use in the fit.
    """
    obs = {}
    obs["filters"], obs["maggies"], obs['maggies_unc'] = load_maggies_from_galaxy(galaxy, snr)
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


def load_maggies_from_galaxy(galaxy, snr):

    # TODO redo unified columns to preserve original filters
    # For now, let's just use a bunch of sdss unified columns
    galaxy_filter_cols = ['u', 'g', 'r', 'i', 'z', 'w1', 'w2', 'w3','w4']

    #  'y', 'j', 'h', 'k',

    # These are the names of the relevant filters, 
    # in the same order as the photometric data (see below)
    sdss = ['sdss_{0}0'.format(b) for b in ['u', 'g', 'r', 'i', 'z']]
    # vvvv = ['y', 'j', 'h', 'k']
    wise = ['wise_{}'.format(x) for x in ['w1', 'w2', 'w3', 'w4']]

    filternames = sdss + wise
    # And here we instantiate the `Filter()` objects using methods in `sedpy`
    filters = sedpy.observate.load_filters(filternames)

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # These should be in apparent AB magnitudes
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    mags = np.array(galaxy[galaxy_filter_cols].values).astype(float)
    maggies = 10**(-0.4*mags)

    # And now we store the uncertainties (again in units of maggies)
    # In this example we are going to fudge the uncertainties based on the supplied `snr` meta-parameter.
    maggies_unc = (1./snr) * maggies

    return filters, maggies, maggies_unc



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




def build_model(redshift=None, fixed_metallicity=None, dust=False, 
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
    model_params = TemplateLibrary["continuity_flex_sfh"]
    
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
        # make sure zred is fixed
        model_params["zred"]['isfree'] = True
    else:
        model_params["zred"]['isfree'] = False
        # And set the value to the redshift keyword
        model_params["zred"]['init'] = redshift

    if dust:
        # Add dust emission (with fixed dust SED parameters)
        # Since `model_params` is a dictionary of parameter specifications, 
        # and `TemplateLibrary` returns dictionaries of parameter specifications, 
        # we can just update `model_params` with the parameters described in the 
        # pre-packaged `dust_emission` parameter set.
        model_params.update(TemplateLibrary["dust_emission"])
        
    # Now instantiate the model object using this dictionary of parameter specifications
    model = SedModel(model_params)

    return model
