#!/bin/bash

# example use: ./repos/agnfinder/submit.sh planet 0 3

export REPO=/mnt/zfsusers/mwalmsley/repos/agnfinder
export PYTHON=/mnt/zfsusers/mwalmsley/envs/agnfitter/bin/python
export QUEUE=$1  # e.g planet

START=$2
END=$3
for INDEX in $(eval echo {$START..$END});
do
    echo "Sampling galaxy $INDEX"
    FILE=${REPO}/results/emulated_sampling/galaxy_${INDEX}_performance.h5
    if test -f "$FILE"; then
        echo "$FILE already exists"
    else
        addqueue -c "10 minutes maybe" -q $QUEUE -n 2 -m 2 $PYTHON ${REPO}/agnfinder/tf_sampling/run_sampler_parallel.py  --index $INDEX --checkpoint-loc $REPO/results/checkpoints/latest --output-dir $REPO/results/emulated_sampling
    fi
done
