
# Oxford Cluster

**Open Oxford cluster**
ping -c 2 -s 999 glamdring.physics.ox.ac.uk

**Sync agnfinder code to Oxford cluster**
rsync -avz --exclude 'results' repos/agnfinder mwalmsley@glamdring.physics.ox.ac.uk:repos

**sync results from Oxford cluster**
rsync -avz mwalmsley@glamdring.physics.ox.ac.uk:repos/agnfinder/results/ /media/mike/internal/agnfinder/results/...

ssh mwalmsley@glamdring.physics.ox.ac.uk

# UCSC Cluster

vpn.ucsc.edu

ssh mwalmsle@lux.ucsc.edu

srun -N 1 --partition=gpuq  --pty bash -i
module load python36 (job-free environment only)
source $tfgpu_env

rsync -avz --exclude 'results' repos/agnfinder mwalmsle@lux.ucsc.edu:repos
rsync -avz mwalmsle@lux.ucsc.edu:repos/agnfinder/results/ /media/mike/internal/agnfinder/results/ucsc_cluster_emulated

# Profiling

pyprof2calltree -k -i myscript.cprof

# Run

python agnfinder/prospector/main.py --index=0 --galaxy=qso
