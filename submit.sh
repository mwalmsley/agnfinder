#!/bin/bash

# example use from repo root: ./submit.sh planet 0 3 1 4

REPO=/mnt/zfsusers/mwalmsley/repos/agnfinder
PYTHON=/mnt/zfsusers/mwalmsley/miniconda3/envs/agnfinder/bin/python
QUEUE=$1  # e.g planet
START=$2
END=$3
NODES=$4
MEMORY=$5  # per node!

for INDEX in $(eval echo {$START..$END});
do
    echo "Sampling galaxy $INDEX"
    FILE=${REPO}/results/emulated_sampling/galaxy_${INDEX}_performance.h5
    # if test -f "$FILE"; then
    #     echo "$FILE already exists"
    # else
    addqueue -c "5 hour production" -q $QUEUE -n $NODES -m $MEMORY $PYTHON $REPO/agnfinder/prospector/main.py --name galaxy --cube "Yes" --save-dir $REPO/results/vanilla_emcee --index $INDEX
    # fi
done

# no point using less than 4gb memory on planet queue