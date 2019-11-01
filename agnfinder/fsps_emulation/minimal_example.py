import logging
import fsps
sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                sfh=0, logzsol=0.0, dust_type=2, dust2=0.2)
print(sp.libraries)

sdss_bands = fsps.find_filter('sdss')
print(sdss_bands)
print(sp.get_mags(tage=13.7, bands=sdss_bands))

sp.params['logzsol'] = -1
print(sp.get_mags(tage=13.7, bands=sdss_bands))

wave, spec = sp.get_spectrum(tage=13.7)
print(sp.formed_mass)
print(sp.stellar_mass)