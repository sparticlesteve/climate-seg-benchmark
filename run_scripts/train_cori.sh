#!/bin/bash
#SBATCH -J climseg-cori
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30

##SBATCH -S 2
##DW persistentdw name=DeepCAM

# Job parameters
do_stage=false
do_train=true
do_test=false
numfiles_train=32
numfiles_validation=32
numfiles_test=0
num_epochs=1
grad_lag=1

# Setup software
module load tensorflow/intel-1.13.1-py36
export OMP_NUM_THREADS=66
# Trying a different OMP config, mainly just to silence verbosity
export KMP_AFFINITY="granularity=fine,compact,1,0"
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread
export MKLDNN_VERBOSE=0 #2 is very verbose

# Setup directories
datadir=/project/projectdirs/dasrepo/gb2018/tiramisu/segm_h5_v3_split
scratchdir=$datadir # no staging
#scratchdir=${DW_PERSISTENT_STRIPED_DeepCAM}/$(whoami)
run_dir=$SCRATCH/climate-seg-benchmark/run_cori/run_n${SLURM_NNODES}_j${SLURM_JOBID}
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
        #cmd="srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
        cmd="srun -N 1 -n 1 -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${numfiles_train} ${numfiles_validation} ${numfiles_test}"
	echo ${cmd}
	${cmd}
    fi
else
    echo "Scratchdir and datadir is the same, no staging needed!"
fi

# Run the training
if $do_train; then
    echo "Starting Training"
    runid=0
    runfiles=$(ls -latr out.lite.fp32.lag${grad_lag}.train.run* | tail -n1 | awk '{print $9}')
    if [ ! -z ${runfiles} ]; then
        runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
    fi

    srun python -u ./deeplab-tf-train.py \
        --datadir_train ${scratchdir}/train \
        --train_size ${numfiles_train} \
        --datadir_validation ${scratchdir}/validation \
        --validation_size ${numfiles_validation} \
        --downsampling 2 \
        --channels 0 1 2 10 \
        --chkpt_dir checkpoint.fp32.lag${grad_lag} \
        --epochs ${num_epochs} \
        --fs global \
        --loss weighted_mean \
        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${grad_lag} \
        --model=resnet_v2_50 \
        --scale_factor 1.0 \
        --batch 1 \
        --decoder=deconv1x \
        --device "/device:cpu:0" \
        --label_id 0 \
        --disable_imsave \
        --tracing="2:5" \
        --trace_dir="./" \
        --data_format "channels_last" |& tee out.lite.fp32.lag${grad_lag}.train.run${runid}
fi

if $do_test; then
    echo "Starting Testing"
    runid=0
    runfiles=$(ls -latr out.lite.fp32.lag${grad_lag}.test.run* | tail -n1 | awk '{print $9}')
    if [ ! -z ${runfiles} ]; then
        runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
    fi

    python -u ./deeplab-tf-inference.py \
        --datadir_test ${scratchdir}/test \
        --test_size ${numfiles_test} \
        --downsampling 2 \
        --channels 0 1 2 10 \
        --chkpt_dir checkpoint.fp32.lag${grad_lag} \
        --output_graph deepcam_inference.pb \
        --output output_test \
        --fs "local" \
        --loss weighted_mean \
        --model=resnet_v2_50 \
        --scale_factor 1.0 \
        --batch 1 \
        --decoder=deconv1x \
        --device "/device:cpu:0" \
        --label_id 0 \
        --data_format "channels_last" |& tee out.lite.fp32.lag${grad_lag}.test.run${runid}
fi
