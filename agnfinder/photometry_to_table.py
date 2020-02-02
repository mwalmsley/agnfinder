import pandas as pd
import numpy as np

from agnfinder.prospector import load_photometry

def load_maggies_fast(galaxy, errors, selection='reliable'):
    # minimal replacement for load_photometry_from_galaxy()
    all_filters = load_photometry.get_filters(selection)
    valid_filters = [f for f in all_filters if load_photometry.filter_has_valid_data(f, galaxy)]
    mags = np.array(galaxy[[f.mag_col for f in valid_filters]].values).astype(float)
    maggies = 10**(-0.4*mags)

    if errors:
        mag_errors = np.array(galaxy[[f.error_col for f in valid_filters]].values).astype(float) * 5.  # being skeptical...
        maggies_unc = np.zeros(len(mags))
        for i in range(len(mags)):
            maggies_unc[i] = load_photometry.calculate_maggie_uncertainty(mag_errors[i], maggies[i]).astype(float)
    else:
        maggies_unc = None
    return valid_filters,  maggies, maggies_unc

def load_valid_photometry_to_series(galaxy, errors=False):
    data = {}
    filters, maggies, maggies_unc = load_maggies_fast(galaxy, errors)
    photometry = dict(zip([f.bandpass_file for f in filters], maggies))
    data.update(photometry)
    if maggies_unc is not None:
        photometry_err = dict(zip([f.bandpass_file + '_err' for f in filters], maggies_unc))
        data.update(photometry_err)
    return pd.Series(data)


def keep_only_reliable_galaxies(df):
    # Exclude galex and kids as often wrong/missing
    # keep_cols = {col for col in phot_df.columns.values}
    reliable_df = df[[col for col in df.columns.values if 'kids' not in col]]
    reliable_df = reliable_df[[col for col in reliable_df.columns.values if 'galex' not in col]]
    reliable_df = reliable_df[[col for col in reliable_df.columns.values if 'cfht' not in col]]
    return reliable_df.dropna()


def get_table(df, reliable, errors):
    rows = []
    for i in range(len(df)):
        rows.append(load_valid_photometry_to_series(df.iloc[i], errors=errors))
    phot_df = pd.DataFrame(rows)
    if reliable:
        return keep_only_reliable_galaxies(phot_df)
    else:
        return phot_df
