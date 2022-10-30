#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=80GB
#PBS -l jobfs=200GB
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -P uo40
#PBS -l walltime=40:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd


module load python3/3.9.2
module load pytorch/1.9.0
cd /scratch/li96/lt2442/Focal-Loss-Search
python3 train_search.py --noCEFormat --load_checkpoints --predictor_warm_up=5000 --num_states=$NUM_STATES --num_obj=$NUM_OBJ --predictor_lambda=$PRED_LDA --lfs_lambda=$LFS_LDA --data=../datasets
