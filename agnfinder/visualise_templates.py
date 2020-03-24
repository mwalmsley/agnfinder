import matplotlib.pyplot as plt
import numpy as np

# https://www.cfht.hawaii.edu/Instruments/Imaging/Megacam/specsinformation.html
# http://skyserver.sdss.org/dr7/en/proj/advanced/color/sdssfilters.asp
# http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/filter-set
# http://www.ukidss.org/technical/photom/hewett-ukidss.pdf
# http://www.astro.ucla.edu/~wright/WISE/passbands.html
# in micrometers
FILTERS = {
    'u_cfht': 0.355,
    'g_cfht': 0.475,
    'r_cfht': 0.640,
    'i_cfht_old': 0.776,
    'i_cfht_new': 0.776,  # allegedly?
    'z_cfht': 0.925,
    'u_sdss': 0.3543,
    'g_sdss': 0.4770,
    'r_sdss': 0.6231,
    'i_sdss': 0.7625,
    'z_sdss': 0.9134,  # fairly different to vista!
    'z_vista': 0.877,
    'Y_vista': 1.020,
    'J_vista': 1.252,
    'H_vista': 1.645,
    'Ks_vista': 2.147,
    'J_ukidss': 1.2483,
    'H_ukidss': 1.6313,
    'K_ukidss': 2.2010,
    'WISE_1': 3.368,
    'WISE_2': 4.618
}

def visualise_mags(template, ax, **kwargs):
    mags = []
    wavelengths = []
    for filter_name, wavelength in FILTERS.items():
        mags.append(template[filter_name])
        wavelengths.append(wavelength)
    # normalise to mean = 0
    normalised_mags = np.array(mags) - np.mean(mags)
    ax.errorbar(x=wavelengths, y=normalised_mags, fmt='.', **kwargs)  # TODO look up band widths
    ax.set_xlabel(r'$\lambda$ ($\mu m$)')
    ax.set_ylabel('Normalised Mag')
