import pytest

import os

import numpy as np
import matplotlib.pyplot as plt

from agnfinder import visualise_templates
from tests import TEST_FIGURE_DIR


@pytest.fixture
def example_sed():
    mag_cols = [
        'u_cfht',
        'g_cfht',
        'r_cfht',
        'i_cfht_old',
        'i_cfht_new',
        'z_cfht',
        'u_sdss', 
        'g_sdss',
        'r_sdss',
        'i_sdss',
        'z_sdss',
        'z_vista',
        'Y_vista',
        'J_vista',
        'H_vista',
        'Ks_vista', 
        'J_ukidss',
        'H_ukidss',
        'K_ukidss',
        'WISE_1',
        'WISE_2'
    ]
    mag_trend = np.linspace(-15., -16., len(mag_cols))
    noise = np.random.normal(loc=0., scale=0.05, size=len(mag_cols))
    return dict(zip(mag_cols, mag_trend + noise))

# def test_visualise_mags(example_sed):
#     fig, ax = plt.subplots()
#     visualise_templates.visualise_mags(example_sed, ax)  # inplace
#     fig.tight_layout()
#     fig.savefig(os.path.join(TEST_FIGURE_DIR, 'visualise_mags.png'))
    