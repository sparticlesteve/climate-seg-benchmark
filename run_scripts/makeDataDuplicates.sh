#!/bin/bash

# This script will make 2 extra copies of every data file in the provided
# directory, adding "copy1" and "copy2" to the filenames.
#
# Usage: makeDataDuplicates.sh path/to/data/train

dir=$1
for f in $dir/ah_data-*[0-9].h5; do
    f1=${f/.h5/-copy1.h5}
    f2=${f/.h5/-copy2.h5}
    echo "copying $f $f1 $f2"
    cp $f $f1
    cp $f $f2
done
