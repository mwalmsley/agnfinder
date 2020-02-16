import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from sedpy import observate
import tqdm
import dill

class Filter():

    def __init__(self, bandpass_file, mag_col, error_col):
        self.bandpass_file = bandpass_file
        self.mag_col = mag_col
        self.mag_error_col = error_col

        self.maggie_col = mag_col.replace('mag', 'maggie')
        self.maggie_error_col = error_col.replace('mag', 'maggie')


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
    
    # note that these are *not* in wavelength order!
    all_filters =  galex + sdss+ cfht + kids + vista + wise

    if selection == 'reliable':
        return sdss + vista + wise
    elif selection == 'euclid':
        return sdss + vista_euclid
    elif selection == 'all':
        return all_filters
    else:
        raise ValueError(f'Filter selection {selection} not recognised')


def add_maggies_cols(input_df):
    # run once on catalog of real galaxies, before doing anything with it
    # assume filled values for all 'reliable' filters
    df = input_df.copy()  # don't modify inplace
    filters = get_filters('reliable')
    for f in tqdm.tqdm(filters):
        df[f.maggie_col] = df[f.mag_col].apply(mags_to_maggies)
        df[f.maggie_error_col] = df[[f.mag_error_col, f.maggie_col]].apply(lambda x: calculate_maggie_uncertainty(*x), axis=1)
    return df


def mags_to_maggies(mags):
    # mags should be apparent AB magnitudes
    # The units of the fluxes need to be maggies (Jy/3631)
    return 10**(-0.4*mags)


def load_maggies_to_array(galaxy, filters):
    maggies = np.array([galaxy[f.maggie_col] for f in filters])
    maggies_unc = np.array([galaxy[f.maggie_error_col] for f in filters])
    return maggies, maggies_unc


def load_galaxy_for_prospector(galaxy, filter_selection):
    all_filters = get_filters(filter_selection)
    valid_filters = [f for f in all_filters if filter_has_valid_data(f, galaxy)]
    if filter_selection == 'reliable' and len(valid_filters) != 12:
        raise ValueError('Some reliable bands are missing - only got {}'.format(valid_filters))
    if filter_selection == 'euclid' and len(valid_filters) != 8:
        raise ValueError('Needs 8 valid Euclid bands - only got {}'.format(valid_filters))
    logging.debug('valid filters: {}'.format(valid_filters))

    maggies, maggies_unc = load_maggies_to_array(galaxy, valid_filters)
    # Instantiate the `Filter()` objects using methods in `sedpy`
    # prospector needs these
    filters = observate.load_filters([f.bandpass_file for f in valid_filters])

    return filters, maggies, maggies_unc

def load_dummy_galaxy_for_prospector(galaxy, filter_selection):
    filters = get_filters(selection=filter_selection)
    loaded_filters = observate.load_filters([f.bandpass_file for f in filters])
    maggies = np.ones(len(filters))
    maggies_unc = np.ones(len(filters))
    return loaded_filters, maggies, maggies_unc


def filter_has_valid_data(filter_tuple, galaxy):  # now looks for maggie cols, not mag cols
    filter_value = galaxy[filter_tuple.maggie_col]
    valid_value = not pd.isnull(filter_value) and filter_value > -98 and filter_value < 98
    filter_error = galaxy[filter_tuple.maggie_error_col]
    valid_error = not pd.isnull(filter_error) and filter_error > 0  # <0 if -99 (unknown) or -1 (only upper bound)
    return valid_value and valid_error


def calculate_maggie_uncertainty(mag_error, maggie):
    # http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html#magnitudes
    return maggie * mag_error / 1.09


def estimate_maggie_uncertainty(true_observation, min_unc=0.001, max_unc=0.15):
    # for cube galaxies, based on fit to real galaxies
    # true_observation needs batch dimension
    bands = ['u_sloan', 'g_sloan', 'r_sloan', 'i_sloan', 'z_sloan', 'VISTA_H','VISTA_J', 'VISTA_Y']  # euclid bands, hardcoded for now
    assert true_observation.shape[1] == len(bands)

    #  uncertainty = true_observation * 0.05  # assume 5% uncertainty on all bands for simulated galaxies
    error_estimators_loc = 'data/error_estimators.pickle'
    with open(error_estimators_loc, 'rb') as f:
        error_estimators = dill.load(f)
    estimated_uncertainty = np.zeros_like(true_observation).astype(np.float32)
    for galaxy_i, galaxy in enumerate(true_observation):
        for band_i, band in enumerate(bands):
            estimated_uncertainty[galaxy_i, band_i] = error_estimators[band](galaxy[band_i])
    # add clipping
    uncertainty = np.min(np.stack([estimated_uncertainty, true_observation * max_unc]), axis=0)  # 1 sigma uncertainty no more than 15%
    uncertainty = np.max(np.stack([uncertainty, true_observation * min_unc]), axis=0)  # no less than 3%
    return uncertainty
