# Deep Learning Climate Segmentation Benchmark

Reference implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## Dataset

The dataset for this benchmark comes from CAM5 [1] simulations and is hosted at
NERSC. The samples are stored in HDF5 files with input images of shape
(768, 1152, 16) and pixel-level labels of shape (768, 1152). The labels have
three target classes (background, atmospheric river, tropical cycline) and were
produced with TECA [2].

The current recommended way to get the data is to use GLOBUS and the following
globus endpoint:

https://app.globus.org/file-manager?origin_id=0b226e2c-4de0-11ea-971a-021304b0cca7&origin_path=%2F

The dataset folder contains a README with some technical description of the
dataset and an All-Hist folder containing all of the data files.

### Previous dataset for ECP Annual Meeting 2019

This is a smaller dataset (~200GB total) available to get things started.
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

## References

1. Wehner, M. F., Reed, K. A., Li, F., Bacmeister, J., Chen, C.-T., Paciorek, C., Gleckler, P. J., Sperber, K. R., Collins, W. D., Gettelman, A., et al.: The effect of horizontal resolution on simulation quality in the Community Atmospheric Model, CAM5. 1, Journal of Advances in Modeling Earth Systems, 6, 980-997, 2014.
2. Prabhat, Byna, S., Vishwanath, V., Dart, E., Wehner, M., Collins, W. D., et al.: TECA: Petascale pattern recognition for climate science, in: International Conference on Computer Analysis of Images and Patterns, pp. 426-436, Springer, 2015b.
