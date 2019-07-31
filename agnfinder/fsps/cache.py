import fsps

if __name__ == '__main__':

    vactoair_flag = False
    compute_vega_mags = False
    zcontinuous=1  # interpolate internally to logzsol metallicity
    ssp = fsps.StellarPopulation(
        compute_vega_mags=compute_vega_mags,
        zcontinuous=zcontinuous,
        vactoair_flag=vactoair_flag
    )

    tage = 1.  # TODO free param
    wave, spec = ssp.get_spectrum(tage=tage, peraa=False)
    print(wave,'wave')
    print(spec, 'spec')

# https://github.com/bd-j/prospector/blob/master/prospect/sources/galaxy_basis.py#L109

# https://github.com/dfm/python-fsps/blob/master/fsps/fsps.py#L582