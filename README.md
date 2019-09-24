# Deep Learning Climate Segmentation Benchmark

Reference implementation for the climate segmentation benchmark, based on the
Exascale Deep Learning for Climate Analytics codebase here:
https://github.com/azrael417/ClimDeepLearn, and the paper:
https://arxiv.org/abs/1810.01993

## How to get the data

Details forthcoming

## How to run the benchmark

Submission scripts are in `run_scripts`.

### Running at NERSC

To submit to the Cori KNL system, do

```bash
cd run_scripts
sbatch -N 64 train_cori.sh
```
