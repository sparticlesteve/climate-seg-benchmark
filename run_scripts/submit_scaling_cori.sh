#!/bin/bash

# Weak scaling duplicated training dataset (9k files)
sbatch -N 1 -q debug train_cori.sh --ntrain 8 --nvalid 0 --epochs 4
sbatch -N 2 -q debug train_cori.sh --ntrain 16 --nvalid 0 --epochs 4
sbatch -N 4 -q debug train_cori.sh --ntrain 32 --nvalid 0 --epochs 4
sbatch -N 8 -q debug train_cori.sh --ntrain 64 --nvalid 0 --epochs 4
sbatch -N 16 -q debug train_cori.sh --ntrain 128 --nvalid 0 --epochs 4
sbatch -N 32 -q debug train_cori.sh --ntrain 256 --nvalid 0 --epochs 4
sbatch -N 64 -q debug train_cori.sh --ntrain 512 --nvalid 0 --epochs 4
sbatch -N 128 -q debug train_cori.sh --ntrain 1024 --nvalid 0 --epochs 4
sbatch -N 256 -q debug train_cori.sh --ntrain 2048 --nvalid 0 --epochs 4
sbatch -N 512 -q debug train_cori.sh --ntrain 4096 --nvalid 0 --epochs 4
sbatch -N 1024 -q regular train_cori.sh --ntrain 8192 --nvalid 0 --epochs 4

# Weak scaling with dummy dataset (matching above config)
sbatch -N 1 -q debug train_cori.sh --ntrain 8 --nvalid 0 --epochs 4 --dummy
sbatch -N 2 -q debug train_cori.sh --ntrain 16 --nvalid 0 --epochs 4 --dummy
sbatch -N 4 -q debug train_cori.sh --ntrain 32 --nvalid 0 --epochs 4 --dummy
sbatch -N 8 -q debug train_cori.sh --ntrain 64 --nvalid 0 --epochs 4 --dummy
sbatch -N 16 -q debug train_cori.sh --ntrain 128 --nvalid 0 --epochs 4 --dummy
sbatch -N 32 -q debug train_cori.sh --ntrain 256 --nvalid 0 --epochs 4 --dummy
sbatch -N 64 -q debug train_cori.sh --ntrain 512 --nvalid 0 --epochs 4 --dummy
sbatch -N 128 -q debug train_cori.sh --ntrain 1024 --nvalid 0 --epochs 4 --dummy
sbatch -N 256 -q debug train_cori.sh --ntrain 2048 --nvalid 0 --epochs 4 --dummy
sbatch -N 512 -q debug train_cori.sh --ntrain 4096 --nvalid 0 --epochs 4 --dummy
sbatch -N 1024 -q regular train_cori.sh --ntrain 8192 --nvalid 0 --epochs 4 --dummy

# This script shows what was run on Cori so far.
# I used weak scaling in the number of training samples up to 64 nodes
# with a single epoch.
# Beyond that, due to the limited dataset size, I kept the number of
# samples fixed at 3072 and scaled in terms of number of epochs.
# I copied the test dataset into the training dataset directory to have
# enough samples.
## Weak scaling number of samples, 1 epoch.
#sbatch -N 1 -q debug train_cori.sh --ntrain 32 --nvalid 0
#sbatch -N 2 -q debug train_cori.sh --ntrain 64 --nvalid 0
#sbatch -N 4 -q debug train_cori.sh --ntrain 128 --nvalid 0
#sbatch -N 8 -q debug train_cori.sh --ntrain 256 --nvalid 0
#sbatch -N 16 -q debug train_cori.sh --ntrain 512 --nvalid 0
#sbatch -N 32 -q regular train_cori.sh --ntrain 1024 --nvalid 0
#sbatch -N 64 -q regular train_cori.sh --ntrain 2048 --nvalid 0
## Large-scale jobs, fixed number of samples but weakly-scaling in epochs.
#sbatch -N 128 -q regular train_cori.sh --ntrain 3072 --nvalid 0 --epochs 1
#sbatch -N 256 -q regular train_cori.sh --ntrain 3072 --nvalid 0 --epochs 2
#sbatch -N 512 -q regular train_cori.sh --ntrain 3072 --nvalid 0 --epochs 4
#sbatch -N 1024 -q regular train_cori.sh --ntrain 3072 --nvalid 0 --epochs 8
