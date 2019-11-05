# AGNFinder
Detect AGN from photometry in XXL data, as Euclid prep.

## Data Required

You will need:
- The catalog of galaxies and associated photometry, which Mike calls `cpz_paper_sample_week3.parquet` (csv also supported)
- The modified/non-default sedpy filter transmission files. **Place these into the sedpy filters folder, under `[your_sedpy_install]/filters`. Mike has these under the folder `CPz_filters`.

## Installation

AGNFinder requires python-fsps, which itself requires FSPS. 
FSPS installation instructions are [here](https://github.com/cconroy20/fsps/blob/master/doc/INSTALL).
Follow these now. Setting the $SPS_HOME environmental variable in the shell from which you run Python is crucial, or the subsequent python-fsps install will fail.

Clone the repo. 

    git clone git@github.com:mwalmsley/agnfinder.git

From one directory level above (your current directory, by default), run

    pip install -r agnfinder/requirements.txt

Note that requirements.txt will install several packages directly from git. See requirements.txt.
If you receive the an error relating to $SPS_HOME during the pip install of python-fsps, check your FSPS install and check that variable is set (`echo $SPS_HOME`).

Finally, still from the directory above:

    pip install -e agnfinder

The `-e` is important if you plan to edit the agnfinder code. You should now be able to import the package.

## Running

To use the HMC emulator:

    python agnfinder/hmc_emulator/main.py 

with optional arguments as per the file. 

Using all default arguments will raise the following error: `ValueError: It looks like the parameter(s) in column(s) 0, 1, 2, 3, 4, 5, 6 have no dynamic range.`. The default args are an oversimplified test case where there aren't enough samples to make a corner plot. Specify the arguments to resolve this.

## Troubleshooting

- `Filter transmission file /data/miniconda3/envs/agnfinder/lib/python3.6/site-packages/sedpy/data/filters/u_sloan.par does not exist!` = you didn't copy the sedpy filters. See above.

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



Template lists:

Ellipticals

-----------

extiction law: 

    none

EB_V:

    0

emission lines: 

    none

templates:

    1 ILBERT2009/Ell1_A_0.sed

    2 ILBERT2009/Ell2_A_0.sed  

    3 ILBERT2009/Ell3_A_0.sed  

    4 ILBERT2009/Ell4_A_0.sed  

    5 ILBERT2009/Ell5_A_0.sed 

    6 ILBERT2009/Ell6_A_0.sed 

    7 ILBERT2009/Ell7_A_0.sed

Spirals

-------

extiction law:

    SMC_prevot.dat

    SB_calzetti.dat

    SB_calzetti_bump1.dat

    SB_calzetti_bump2.dat

EB_V:

    0.000,0.050,0.100,0.150,0.200,0.250,0.300,0.400,0.500

emission lines: 

    yes

templates:

    1 ILBERT2009/S0_A_0.sed

    2 ILBERT2009/Sa_A_0.sed

    3 ILBERT2009/Sa_A_1.sed

    4 ILBERT2009/Sb_A_0.sed

    5 ILBERT2009/Sb_A_1.sed

    6 ILBERT2009/Sc_A_0.sed

    7 ILBERT2009/Sc_A_1.sed

    8 ILBERT2009/Sc_A_2.sed

    9 ILBERT2009/Sd_A_0.sed

    10 ILBERT2009/Sd_A_1.sed

    11 ILBERT2009/Sd_A_2.sed

Starburst

---------

extiction law: 

    SMC_prevot.dat

    SB_calzetti.dat

    SB_calzetti_bump1.dat

    SB_calzetti_bump2.dat

EB_V:

    0.000,0.050,0.100,0.150,0.200,0.250,0.300,0.400,0.500

emission lines: 

    yes

templates:

    1 ILBERT2009/Sdm_A_0.sed   

    2 ILBERT2009/SB0_A_0.sed   

    3 ILBERT2009/SB1_A_0.sed   

    4 ILBERT2009/SB2_A_0.sed   

    5 ILBERT2009/SB3_A_0.sed   

    6 ILBERT2009/SB4_A_0.sed   

    7 ILBERT2009/SB5_A_0.sed   

    8 ILBERT2009/SB6_A_0.sed   

    9 ILBERT2009/SB7_A_0.sed   

    10 ILBERT2009/SB8_A_0.sed   

    11 ILBERT2009/SB9_A_0.sed   

    12 ILBERT2009/SB10_A_0.sed  

    13 ILBERT2009/SB11_A_0.sed

AGN

---

extiction law: 

    SMC_prevot.dat

EB_V:

    0.000,0.050,0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500

emission lines: 

    none (included in the templates)

templates:

    1 SALVATO2009/M82_template_norm.sed

    2 SALVATO2009/I22491_template_norm.sed

    3 SALVATO2009/Sey18_template_norm.sed

    4 SALVATO2009/Sey2_template_norm.sed

    5 SALVATO2009/S0_90_QSO2_10.sed

    6 SALVATO2009/S0_80_QSO2_20.sed

    7 SALVATO2009/S0_70_QSO2_30.sed

    8 SALVATO2009/S0_60_QSO2_40.sed

    9 SALVATO2009/S0_50_QSO2_50.sed

    10 SALVATO2009/S0_40_QSO2_60.sed

    11 SALVATO2009/S0_30_QSO2_70.sed

    12 SALVATO2009/S0_20_QSO2_80.sed

    13 SALVATO2009/S0_10_QSO2_90.sed

    14 SALVATO2009/Mrk231_template_norm.sed

    15 SALVATO2009/I22491_90_TQSO1_10.sed

    16 SALVATO2009/I22491_80_TQSO1_20.sed

    17 SALVATO2009/I22491_70_TQSO1_30.sed

    18 SALVATO2009/I22491_60_TQSO1_40.sed

    19 SALVATO2009/I22491_50_TQSO1_50.sed

QSO

---

extiction law: 

    SMC_prevot.dat

EB_V:

    0.000,0.050,0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500

emission lines: 

    none (included in the templates)

templates:

    1 QSO/pl_HST_COS_QSO_DR2_029_t0.sed

    2 QSO/pl_HST_COS_QSOH_template_norm.sed

    3 QSO/pl_HST_COS_TQSO1_template_norm.sed

    4 QSO/pl_HST_COS_QSO1_template_norm.sed

    5 QSO/pl_HST_COS_BQSO1_template_norm.sed

    6 QSO/pl_HST_COS_radio_loud_mean_sedMR.sed

    7 QSO/pl_HST_COS_radio_quiet_mean_sedMR.sed
