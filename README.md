# AGNFinder
Detect AGN from photometry in XXL data, as Euclid prep.

## Installation

Clone the repo. 

    git clone git@github.com:mwalmsley/agnfinder.git

From one directory level above (your current directory, by default), run

    pip install -r agnfinder/requirements.txt
    pip install -e agnfinder

You can now import the package.

## Data
XXL LePhare libraries:
- no emission lines, no extinction laws
LIBRARY_XXLN_Ellipticals.lib.dat.fits

- no emission lines, with extinction laws
LIBRARY_XXLN_AGN.lib.dat.fits
LIBRARY_XXLN_QSO.lib.dat.fits

- with emission lines
LIBRARY_XXLN_Spirals.lib.dat.fits
LIBRARY_XXLN_Starburst.lib.dat.fits

CPz Data:
uK_IR_final.fits

