#!/bin/bash
#SBATCH -J climseg-cgpu
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -o %x-%j.out

# Job parameters
rankspernode=8
do_stage=true
do_train=true
do_test=false
numfiles_train=-1
numfiles_validation=-1
numfiles_test=0
num_epochs=32
grad_lag=1
prec=16
batch=2
scale_factor=0.1

# Software setup
module load gcc/7.3.0
module load cuda/10.1.168
module load mpich/3.3.1-debug
module load tensorflow/gpu-1.14.0-py37
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE

# Setup directories
datadir=/project/projectdirs/mpccc/tkurth/DataScience/gb2018/data/segm_h5_v3_new_split_maeve
#datadir=/project/projectdirs/dasrepo/gb2018/tiramisu/segm_h5_v3_split
scratchdir=${datadir} # no staging
run_dir=$SCRATCH/climate-seg-benchmark/run_cgpu
rm -rf ${run_dir}/*
mkdir -p ${run_dir}

# Prepare the run directory
cp stage_in_parallel.sh ${run_dir}/
cp ../utils/parallel_stagein.py ${run_dir}/
cp ../utils/graph_flops.py ${run_dir}/
cp ../utils/tracehook.py ${run_dir}/
cp ../utils/common_helpers.py ${run_dir}/
cp ../utils/data_helpers.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-train.py ${run_dir}/
cp ../deeplab-tf/deeplab-tf-inference.py ${run_dir}/
cp ../deeplab-tf/deeplab_model.py ${run_dir}/
cd ${run_dir}

# Stage data if relevant
if [ "${scratchdir}" != "${datadir}" ]; then
    if $do_stage; then
        cmd="srun --mpi=pmi2 -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 80 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
        echo ${cmd}
        ${cmd}
    fi
else
    echo "Scratchdir and datadir is the same, no staging needed!"
fi

# Run the training
sruncmd="srun -u --mpi=pmi2 -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"
if $do_train; then
    echo "Starting Training"
    ${sruncmd} python -u ./deeplab-tf-train.py \
        --datadir_train ${scratchdir}/train \
        --train_size ${numfiles_train} \
        --datadir_validation ${scratchdir}/validation \
        --validation_size ${numfiles_validation} \
        --chkpt_dir checkpoint.fp${prec}.lag${grad_lag} \
        --disable_checkpoint \
        --epochs $num_epochs \
        --fs global \
        --loss weighted \
        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${grad_lag} \
        --model "resnet_v2_50" \
        --scale_factor ${scale_factor} \
        --batch ${batch} \
        --decoder "deconv1x" \
        --device "/device:cpu:0" \
        --dtype "float${prec}" \
        --label_id 0 \
        --data_format "channels_first" |& tee out.fp${prec}.lag${grad_lag}.train
fi

if $do_test; then
    echo "Starting Testing"
    ${sruncmd} python -u ./deeplab-tf-inference.py \
        --datadir_test ${scratchdir}/test \
        --chkpt_dir checkpoint.fp${prec}.lag${grad_lag} \
        --test_size ${numfiles_test} \
        --output_graph deepcam_inference.pb \
        --output output_test_5 \
        --fs "local" \
        --loss weighted \
        --model "resnet_v2_50" \
        --scale_factor ${scale_factor} \
        --batch ${batch} \
        --decoder "deconv1x" \
        --device "/device:cpu:0" \
        --dtype "float${prec}" \
        --label_id 0 \
        --data_format "channels_last" |& tee out.fp${prec}.lag${grad_lag}.test
fi
