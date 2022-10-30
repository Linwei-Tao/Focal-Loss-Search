#!/bin/bash

ssh lt2442@gadi.nci.org.au 'cd /scratch/li96/lt2442/Focal-Loss-Search/hpc_scripts/Gadi/;
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=2,LFS_LDA=2 search_retrain.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=5,LFS_LDA=5 search_retrain.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=25,LFS_LDA=25 search_retrain.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=2,LFS_LDA=2 search_retrain.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=5,LFS_LDA=5 search_retrain.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=25,LFS_LDA=25 search_retrain.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=2,LFS_LDA=2 search_retrain_noCE.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=5,LFS_LDA=5 search_retrain_noCE.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain_noCE.sh
qsub -v NUM_STATES=20,NUM_OBJ=2,PRED_LDA=25,LFS_LDA=25 search_retrain_noCE.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=2,LFS_LDA=2 search_retrain_noCE.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=5,LFS_LDA=5 search_retrain_noCE.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=10,LFS_LDA=10 search_retrain_noCE.sh
qsub -v NUM_STATES=14,NUM_OBJ=2,PRED_LDA=25,LFS_LDA=25 search_retrain_noCE.sh
'