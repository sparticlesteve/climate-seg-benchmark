#!/bin/bash

# Print the header
echo "n_workers train_time n_steps batch_size peak_flops sust_flops"

# Loop over the provided log files
for log_file in $@; do

    # Extract number of workers
    n_workers=$(grep "Num workers" $log_file | awk '{print $3}')
    
    # Extract time
    train_time=$(grep "COMPLETED: training" $log_file | tail -n 1 | awk '{print $12}' | tr -d ',')

    # Extract the number of steps
    n_steps=$(grep REPORT $log_file | wc -l)

    # Extract batch size
    local_batch_size=$(grep "Local batch size" $log_file | awk '{print $4}')
    batch_size=$(( $local_batch_size * $n_workers ))
    
    # Extract peak flops
    peak_flops=$(grep REPORT $log_file | tail -n 1 | awk '{print $16}' | tr -d ',')
    
    # Extract sustained flops
    sust_flops=$(grep "COMPLETED: training" $log_file | tail -n 1 | awk '{print $14}')
    
    # Print out the results
    echo "$n_workers $train_time $n_steps $batch_size $peak_flops $sust_flops"

done
