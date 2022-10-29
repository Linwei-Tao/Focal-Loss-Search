#!/bin/bash
#PBS -P ContraGAN
#PBS -l select=1:ncpus=9:ngpus=1:mem=30GB
#PBS -l walltime=20:00:00
#PBS -j oe

#Load modules
module load python/3.7.7 magma/2.5.3 openmpi-gcc/3.1.5

virtualenv --system-site-packages ~/pytorch
source ~/pytorch/bin/activate

pip install /usr/local/pytorch/cuda10.2/torch-1.10.1-cp37-cp37m-linux_x86_64.whl
pip install /usr/local/pytorch/cuda10.2/torchvision-0.11.0a0+3a7e5e3-cp37-cp37m-linux_x86_64.whl

cd "/scratch/ContraGAN/projects/Focal-Loss-Search"
python3 train_search.py --num_states=$NUM_STATES --num_obj=$NUM_OBJ --predictor_lambda=$PRED_LDA --lfs_lambda=$LFS_LDA --data=../datasets