
# Glamdring

ping -c 2 -s 999 glamdring.physics.ox.ac.uk
ssh mwalmsley@glamdring.physics.ox.ac.uk
q
n
watch tail -30 /var/logs


**Sync agnfinder code to Glamdring**

rsync -avz /home/walml/anaconda3/envs/agnfinder/lib/python3.7/site-packages/sedpy/data/filters/ mwalmsley@glamdring.physics.ox.ac.uk:/mnt/zfsusers/mwalmsley/miniconda3/envs/agnfinder/lib/python3.7/site-packages/sedpy/data/filters

rsync -avz --exclude 'results' repos/agnfinder mwalmsley@glamdring.physics.ox.ac.uk:repos
rsync -avz repos/agnfinder/results/checkpoints mwalmsley@glamdring.physics.ox.ac.uk:repos/agnfinder/results


**sync results from Glamdring**
rsync -avz mwalmsley@glamdring.physics.ox.ac.uk:repos/agnfinder/results/ /media/mike/internal/agnfinder/results/...


# Profiling

pyprof2calltree -k -i myscript.cprof

# Run

python agnfinder/prospector/main.py --index=0 --galaxy=qso

python agnfinder/lfi/train.py data/photometry_simulation_100000.hdf5 --test

export REPO=/mnt/zfsusers/mwalmsley/repos/agnfinder
export PYTHON=/mnt/zfsusers/mwalmsley/miniconda3/envs/agnfinder/bin/python
export QUEUE=planet
export CATALOG=/mnt/zfsusers/mwalmsley/repos/agnfinder/data/cpz_paper_sample_week3_maggies.parquet

addqueue -c "1 hour" -q $QUEUE -n 12 -m 3 $PYTHON $REPO/agnfinder/prospector/main.py cube_test --cube $REPO/data/cubes/latest --save-dir $REPO/results/vanilla_mcmc
addqueue -c "1 hour" -q $QUEUE -n 12 -m 3 $PYTHON $REPO/agnfinder/simulation_samples.py 10000 --catalog-loc data/cpz_paper_sample_week3.parquet

scp mwalmsley@glamdring.physics.ox.ac.uk:/mnt/zfsusers/mwalmsley/repos/agnfinder/results/latest_posterior_stripes.png results/...

## Zeus

ssh mikewalmsley@aquila.star.bris.ac.uk
ssh mike@zeus.star.bris.ac.uk



scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk mike@zeus.star.bris.ac.uk:/scratch/agnfinder/agnfinder/results/latest_posterior_stripes.png results/...

scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk /media/mike/beta/agnfinder/CPz_filters mike@zeus.star.bris.ac.uk:/scratch/agnfinder/data/CPz_filters
cp /scratch/agnfinder/data/CPz_filters/* /home/mike/.conda/envs/agnfinder/lib/python3.7/site-packages/sedpy/data/filters

scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk mike@zeus.star.bris.ac.uk:/scratch/agnfinder/results/emulated_sampling/latest_80000_512_optimised results/emulated_sampling/standard_repeats

scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk mike@zeus.star.bris.ac.uk:/scratch/agnfinder/agnfinder/results/hyperband/agnfinder_4layer_dropout  results/hyperband/latest

scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk data/photometry_quicksave.parquet mike@zeus.star.bris.ac.uk:/scratch/agnfinder/agnfinder/data/photometry_quicksave.parquet

scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk mike@zeus.star.bris.ac.uk:/scratch/agnfinder/agnfinder/data/cubes/x_test_v2.npy data/cubes/
scp -r -oProxyJump=mikewalmsley@aquila.star.bris.ac.uk mike@zeus.star.bris.ac.uk:/scratch/agnfinder/agnfinder/data/cubes/y_test_v2.npy data/cubes/

## ARC

ssh -X chri5177@oscgate.arc.ox.ac.uk
ssh -X arcus-htc

sbatch script.sh
squeue -A phys-zooniverse
sacct




# UCSC (deprecated)

vpn.ucsc.edu

ssh mwalmsle@lux.ucsc.edu

srun -N 1 --partition=gpuq  --pty bash -i
module load python36 (job-free environment only)
source $tfgpu_env

rsync -avz --exclude 'results' repos/agnfinder mwalmsle@lux.ucsc.edu:repos
rsync -avz mwalmsle@lux.ucsc.edu:repos/agnfinder/results/lfi/ /media/mike/internal/agnfinder/results/lfi/


docker run -u $(id -u):$(id -g) --gpus all -it tensorflow/tensorflow:latest-gpu-py3
docker run -u $(id -u):$(id -g) --gpus all -it agnfinder:latest
