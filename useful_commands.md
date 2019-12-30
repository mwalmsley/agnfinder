
# Oxford Cluster

**Open Oxford cluster**
ping -c 2 -s 999 glamdring.physics.ox.ac.uk

**Sync agnfinder code to Oxford cluster**
rsync -avz --exclude 'results' repos/agnfinder mwalmsley@glamdring.physics.ox.ac.uk:repos
rsync -avz repos/agnfinder/results/checkpoints mwalmsley@glamdring.physics.ox.ac.uk:repos/agnfinder/results


**sync results from Oxford cluster**
rsync -avz mwalmsley@glamdring.physics.ox.ac.uk:repos/agnfinder/results/ /media/mike/internal/agnfinder/results/...

ssh mwalmsley@glamdring.physics.ox.ac.uk
q
n

watch tail -30 /var/logs

# UCSC Cluster

vpn.ucsc.edu

ssh mwalmsle@lux.ucsc.edu

srun -N 1 --partition=gpuq  --pty bash -i
module load python36 (job-free environment only)
source $tfgpu_env

rsync -avz --exclude 'results' repos/agnfinder mwalmsle@lux.ucsc.edu:repos
rsync -avz mwalmsle@lux.ucsc.edu:repos/agnfinder/results/lfi/ /media/mike/internal/agnfinder/results/lfi/

# Profiling

pyprof2calltree -k -i myscript.cprof

# Run

python agnfinder/prospector/main.py --index=0 --galaxy=qso

python agnfinder/lfi/train.py data/photometry_simulation_100000.hdf5 --test

export REPO=/mnt/zfsusers/mwalmsley/repos/agnfinder
export PYTHON=/mnt/zfsusers/mwalmsley/envs/agnfitter/bin/python
export QUEUE=planet
export CATALOG=/mnt/zfsusers/mwalmsley/repos/agnfinder/data/cpz_paper_sample_week3.parquet
addqueue -c "1 hour" -q $QUEUE -n 12 -m 3 $PYTHON $REPO/agnfinder/simulation_samples.py 10000 --catalog-loc data/cpz_paper_sample_week3.parquet