#!/bin/bash

# example use: ./repos/agnfinder/submit.sh planet 0 3

REPO=/mnt/zfsusers/mwalmsley/repos/agnfinder
PYTHON=/mnt/zfsusers/mwalmsley/envs/agnfitter/bin/python
QUEUE=$1  # e.g planet
START=$2
END=$3
NODES=$4
MEMORY=$5  # per node!
SAMPLES=$6
INIT=$7
for INDEX in $(eval echo {$START..$END});
do
    echo "Sampling galaxy $INDEX"
    FILE=${REPO}/results/emulated_sampling/galaxy_${INDEX}_performance.h5
    if test -f "$FILE"; then
        echo "$FILE already exists"
    else
        addqueue -c "15 minutes" -q $QUEUE -n $NODES -m $MEMORY $PYTHON ${REPO}/agnfinder/tf_sampling/run_sampler_parallel.py  --index $INDEX --n-samples $SAMPLES --checkpoint-loc $REPO/results/checkpoints/10m_3_epochs_p98 --output-dir $REPO/results/emulated_sampling
    fi
done

# no point using less than 4gb memory on planet queue