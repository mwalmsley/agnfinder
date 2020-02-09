import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from sedpy import observate

Filter = namedtuple('Filter', ['bandpass_file', 'mag_col', 'error_col'])



def get_filters(selection):
        # Pairs of (filter name in sedpy, filter name in dataframe)
    galex = [
        Filter(
            bandpass_file='{}_galex'.format(b),
            mag_col='mag_auto_galex_{}_dr67'.format(b.lower()),
            error_col='magerr_auto_galex_{}_dr67'.format(b.lower())
        )
        for b in ['NUV', 'FUV']]
    # cfht awkward due to i filter renaming - for now, am using i=i_new
    cfht = [
        Filter(
            bandpass_file='{}_cfhtl'.format(b),
            mag_col='mag_auto_cfhtwide_{}_dr7'.format(b),
            error_col='magerr_auto_cfhtwide_{}_dr7'.format(b)
        )
        for b in ['g', 'i', 'r', 'u', 'z']]
    kids = [
        Filter(
            bandpass_file='{}_kids'.format(b),
            mag_col='mag_auto_kids_{}_dr2'.format(b),
            error_col='magerr_auto_kids_{}_dr2'.format(b))
        for b in ['i', 'r']]
    vista = [
        Filter(
            bandpass_file='VISTA_{}'.format(b),
            mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
            error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s'))
        )
        for b in ['H', 'J', 'Ks', 'Y', 'Z']]  # is k called ks in df? TODO
    vista_euclid = [  # only these plus sloan will be available for euclid
        Filter(
            bandpass_file='VISTA_{}'.format(b),
            mag_col='mag_auto_viking_{}_dr2'.format(b.lower().strip('s')),
            error_col='magerr_auto_viking_{}_dr2'.format(b.lower().strip('s'))
        )
        for b in ['H', 'J', 'Y']]  # is k called ks in df? TODO
    sdss = [
        Filter(
            bandpass_file='{}_sloan'.format(b),
            mag_col='mag_auto_sdss_{}_dr12'.format(b),
            error_col='magerr_auto_sdss_{}_dr12'.format(b))
        for b in ['u', 'g', 'r', 'i', 'z']]
    wise = [
        Filter(
            bandpass_file='wise_{}'.format(x),
            mag_col='mag_auto_AllWISE_{}'.format(x.upper()),
            error_col='magerr_auto_AllWISE_{}'.format(x.upper())
        )
        for x in ['w1', 'w2']] # exclude w3, w4
    
    if selection == 'reliable':
        return sdss + vista + wise
    elif selection == 'euclid':
        return sdss + vista_euclid
    elif selection == 'all':
        return galex + sdss+ cfht + kids + vista + wise  # note that these are *not* in wavelength order!
    else:
        raise ValueError(f'Filter selection {selection} not recognised')


def load_maggies_from_galaxy(galaxy, filter_selection):
    all_filters = get_filters(filter_selection)
    valid_filters = [f for f in all_filters if filter_has_valid_data(f, galaxy)]
    if filter_selection == 'reliable' and len(valid_filters) != 12:
        raise ValueError('Some reliable bands are missing - only got {}'.format(valid_filters))
    if filter_selection == 'euclid' and len(valid_filters) != 8:
        raise ValueError('Needs 8 valid Euclid bands - only got {}'.format(valid_filters))
    logging.debug('valid filters: {}'.format(valid_filters))

    # Instantiate the `Filter()` objects using methods in `sedpy`
    filters = observate.load_filters([f.bandpass_file for f in valid_filters])

    # Now we store the measured fluxes for a single object, **in the same order as "filters"**
    # These should be in apparent AB magnitudes
    # The units of the fluxes need to be maggies (Jy/3631) so we will do the conversion here too.
    mags = np.array(galaxy[[f.mag_col for f in valid_filters]].values).astype(float)
    logging.debug('magnitudes: {}'.format(mags))
    maggies = 10**(-0.4*mags)
    logging.debug('maggies: {}'.format(maggies))

    # TODO review error scaling, noise model, lnprobfn - currently a big gap in understanding!
    mag_errors = np.array(galaxy[[f.error_col for f in valid_filters]].values).astype(float)
    logging.debug('mag errors: {}'.format(mag_errors))

    maggies_unc = []
    for i in range(len(mags)):
        maggies_unc.append(calculate_maggie_uncertainty(mag_errors[i], maggies[i]))
    maggies_unc = np.array(maggies_unc).astype(float)
    logging.debug('maggis errors: {}'.format(maggies_unc))

    return filters, maggies, maggies_unc


def filter_has_valid_data(filter_tuple, galaxy):
    filter_value = galaxy[filter_tuple.mag_col]
    valid_value = not pd.isnull(filter_value) and filter_value > -98 and filter_value < 98
    filter_error = galaxy[filter_tuple.error_col]
    valid_error = not pd.isnull(filter_error) and filter_error > 0  # <0 if -99 (unknown) or -1 (only upper bound)
    return valid_value and valid_error


def calculate_maggie_uncertainty(mag_error, maggie):
    # http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html#magnitudes
    return maggie * mag_error / 1.09
