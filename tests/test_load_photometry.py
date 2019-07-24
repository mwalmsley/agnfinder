import pytest

import numpy as np
import pandas as pd

from agnfinder.prospector import load_photometry


@pytest.fixture()
def galaxy():
    return pd.Series(
        {
            # band a is good
            'band_a': 12.,
            'band_a_err': 1.,
            # band b has bad error (upper bound)
            'band_b': 13.,
            'band_b_err': -1.,
            # band c has bad value (nan)
            'band_c': np.nan,
            'band_c_err': 1.,
            # band d has bad value (-99)
            'band_d': -99,
            'band_d_err': 1.,
            # band e has bad error (nan)
            'band_e': 14.,
            'band_e_err': np.nan,
            # band f has bad error (-99)
            'band_f': 14.,
            'band_f_err': -99.
        }
    )

@pytest.fixture()
def filter_tuples():
    tuples = []
    for band in ['a', 'b', 'c', 'd', 'e', 'f']:
        tuples.append(
            load_photometry.Filter(
                bandpass_file='{}.par'.format(band),
                mag_col='band_{}'.format(band),
                error_col='band_{}_err'.format(band)
            )
        )
    return tuples

def test_get_filters():
    filters = load_photometry.get_filters()  # assume okay if throws no errors


def test_filter_has_valid_data(filter_tuples, galaxy):
    validity = [load_photometry.filter_has_valid_data(f, galaxy) for f in filter_tuples]
    assert validity == [True, False, False, False, False, False]
