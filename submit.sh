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
    # only used to check if file already exists
    FILE=${REPO}/results/vanilla_emcee/galaxy_${INDEX}_*.h5
    # https://stackoverflow.com/questions/6363441/check-if-a-file-exists-with-wildcard-in-shell-script
    for f in $FILE; do
        if [ -e "$f" ]; then
            echo "$f already exists, skipping"
        else
            echo "$f not found, beginning sampling"
            addqueue -c "5 hour final sample" -q $QUEUE -n $NODES -m $MEMORY $PYTHON $REPO/agnfinder/prospector/main.py --name galaxy --cube "Yes" --save-dir $REPO/results/vanilla_emcee --index $INDEX
        fi
        break
    done
done

# no point using less than 4gb memory on planet queue