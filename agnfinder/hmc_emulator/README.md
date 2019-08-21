## Create dataset of simulations to emulate

You will need a hypercube of (parameters -> photometry). 

- Download this from Slack - search `photometry_simulations_1000000.hdf5`, or
- Create with `simulation_samples.py` (requires FSPS and a few hours).

## Train and Save Emulator

### Divide into train/test sets

TODO currently this is really hacky, just picks a particular index out as the test case.
Already done for you if you cloned the repo - `lfi_test_case.json`

### Create Emulator

Currently, this happens automatically in the step below, where you can either create a new emulator or use the latest one.

This should probably be refactored...

## (For Test Case) Sample Emulator Quickly

Make sure you've placed the hypercube into `data/photometry_simulaton_1000000.hdf5`, or this will fail.

To train a new emulator, then sample it with default chain lengths/etc:

    main.py --new-emulator

These defauls are designed as a quick debug to make sure nothing fails loudly (if the corner plot itself complains about 'dynamic range', ignore this - it just needs more samples).

After the first run, you'll probably prefer to use the saved emulator and set the HMC arguments to more reasonable values:

    main.py --n-chains=128 --n-burnin=3000 --n-samples=3000

Corner plots are placed in the `results` dir, where the filename will record the optional arguments above. This also needs refactoring.
