## Create dataset of simulations to emulate

You will need a hypercube of (parameters -> photometry). 

- Download this from Slack - search `photometry_simulations_1000000.hdf5`, or
- Create with `simulation_samples.py` (requires FSPS and a few hours).

## Design NN Emulator (Optional)

Use deep_emulator.py to experiment with training NN's, compare to Boosted Trees, and optimise the NN architecture (using hyperopt).
This is optional - for AGNFinder, I've already done some searching for an archicture that does fairly well. This is the default.
If you want to play around with this, you'll need to update the data() function in deep_emulator.py to point to your hypercube.

## Create Emulator and Run on Fixed Test Case

Make sure you've placed the hypercube into `data/photometry_simulaton_1000000.hdf5` or modified data() in `deep_emulator.py§. Else, this will fail.

To train a new emulator, then sample it with default chain lengths/etc on a fixed synthetic galaxy from the test set:

    main.py --new-emulator

These defauls are designed as a quick debug to make sure nothing fails loudly.

After the first run, you'll probably prefer to use the saved emulator and set the HMC arguments to more reasonable values:

    main.py --n-chains=128 --n-burnin=3000 --n-samples=3000

Corner plots are placed in the `results` dir, where the filename will record the optional arguments above. 

## Run Emulator + HMC on Many Galaxies

First, check that the emulator you've trained is actually performing well. Do this with `evaluate_emulator.py`.

Once you're happy the emulator is running well, use `run_sampler.py` to run it on many galaxies. Analyse the results with `evaluate_performance.py`.
