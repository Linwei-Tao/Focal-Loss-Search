#!/bin/bash

# train with focal
ssh ltao0358@hpc.sydney.edu.au 'cd /scratch/ContraGAN/projects/Focal-Loss-Search/hpc_scripts/USYD-HPC-Scripts/;
python3  --num_states=$NUM_STATES --num_obj=$NUM_OBJ --predictor_lambda=$PRED_LDA --lfs_lambda=$LFS_LDA
qsub -v NUM_STATES=12,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain.sh
qsub -v NUM_STATES=12,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain.sh
'