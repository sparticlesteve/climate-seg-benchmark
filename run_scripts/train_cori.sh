#!/bin/bash
#SBATCH -J climseg-cori
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -d singleton
#SBATCH -o %x-%j.out

##SBATCH -S 2
##DW persistentdw name=DeepCAM

# Job parameters
do_stage=false
ntrain=-1
nvalid=-1
ntest=0
batch=1
epochs=1
grad_lag=1
scale_factor=1.0

# Parse command line options
while (( "$#" )); do
    case "$1" in
        --ntrain)
            ntrain=$2
            shift 2
            ;;
        --nvalid)
            nvalid=$2
            shift 2
            ;;
        --ntest)
            ntest=$2
            shift 2
            ;;
        -*|--*=)
            echo "Error: Unsupported flag $1" >&2
            exit 1
            ;;
    esac
done

# Setup software
module load tensorflow/intel-1.13.1-py36
export OMP_NUM_THREADS=66
export KMP_AFFINITY="granularity=fine,compact,1,0"
export MKLDNN_VERBOSE=0 #2 is very verbose
export HDF5_USE_FILE_LOCKING=FALSE

# Setup directories
datadir=/project/projectdirs/dasrepo/gsharing/climseg-benchmark/climseg-data-small
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
        # Multi-node staging wasn't working for me.
        #cmd="srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${ntrain} ${nvalid} ${ntest}"
        cmd="srun -N 1 -n 1 -c 264 ./stage_in_parallel.sh ${datadir} ${scratchdir} ${ntrain} ${nvalid} ${ntest}"
	echo ${cmd}
	${cmd}
    fi
else
    echo "Scratchdir and datadir is the same, no staging needed!"
fi

# Run the training
if [ $ntrain -ne 0 ]; then
    echo "Starting Training"
    runid=0
    runfiles=$(ls -latr out.lite.fp32.lag${grad_lag}.train.run* | tail -n1 | awk '{print $9}')
    if [ ! -z ${runfiles} ]; then
        runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
    fi

    srun python -u ./deeplab-tf-train.py \
        --datadir_train ${scratchdir}/train \
        --train_size ${ntrain} \
        --datadir_validation ${scratchdir}/validation \
        --validation_size ${nvalid} \
        #--downsampling 2 \
        #--channels 0 1 2 10 \
        --chkpt_dir checkpoint.fp32.lag${grad_lag} \
        --epochs ${epochs} \
        --fs global \
        --loss weighted_mean \
        --optimizer opt_type=LARC-Adam,learning_rate=0.0001,gradient_lag=${grad_lag} \
        --model=resnet_v2_50 \
        --scale_factor $scale_factor \
        --batch $batch \
        --decoder=deconv1x \
        --device "/device:cpu:0" \
        --label_id 0 \
        --disable_imsave \
        --tracing="2:5" \
        --trace_dir="./" \
        --data_format "channels_last" |& tee out.lite.fp32.lag${grad_lag}.train.run${runid}
fi

if [ $ntest -ne 0 ]; then
    echo "Starting Testing"
    runid=0
    runfiles=$(ls -latr out.lite.fp32.lag${grad_lag}.test.run* | tail -n1 | awk '{print $9}')
    if [ ! -z ${runfiles} ]; then
        runid=$(echo ${runfiles} | awk '{split($1,a,"run"); print a[1]+1}')
    fi

    python -u ./deeplab-tf-inference.py \
        --datadir_test ${scratchdir}/test \
        --test_size ${ntest} \
        #--downsampling 2 \
        #--channels 0 1 2 10 \
        --chkpt_dir checkpoint.fp32.lag${grad_lag} \
        --output_graph deepcam_inference.pb \
        --output output_test \
        --fs "local" \
        --loss weighted_mean \
        --model=resnet_v2_50 \
        --scale_factor $scale_factor \
        --batch $batch \
        --decoder=deconv1x \
        --device "/device:cpu:0" \
        --label_id 0 \
        --data_format "channels_last" |& tee out.lite.fp32.lag${grad_lag}.test.run${runid}
fi
