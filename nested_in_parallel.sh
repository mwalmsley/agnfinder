#!/bin/sh
SAVE_DIR=results/vanilla_nested
CLASS=passive

CORES=15

LASTCORE=0
for CORE in $(seq 0 1 $CORES)
do
    echo "Starting nested sampling for core $CORE"
    nohup python agnfinder/prospector/main.py $CLASS --index $CORE --catalog-loc data/uk_ir_selection_577.parquet --save-dir $SAVE_DIR --forest $CLASS> "$SAVE_DIR/$CORE.log" &
    LASTCORE=$Z
done