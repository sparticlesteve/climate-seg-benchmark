# Deep Learning Climate Segmentation Benchmark

Reference implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## How to get the data

For now there is a smaller dataset (~200GB total) available to get things started.
It is hosted via Globus:

https://app.globus.org/file-manager?origin_id=bf7316d8-e918-11e9-9bfc-0a19784404f4&origin_path=%2F

and also available via https:

https://portal.nersc.gov/project/dasrepo/deepcam/climseg-data-small/

## How to run the benchmark

Submission scripts are in `run_scripts`.

### Running at NERSC

To submit to the Cori KNL system, do

```bash
# This example runs on 64 nodes.
cd run_scripts
sbatch -N 64 train_cori.sh
```

To submit to the Cori GPU system, do

```bash
# 8 ranks per node, 1 per GPU
module purge
module load esslurm
cd run_scripts
sbatch -N 4 train_corigpu.sh
```
