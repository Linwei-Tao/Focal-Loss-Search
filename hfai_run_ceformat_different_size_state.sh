#!/bin/bash

export MODEL="resnet50"
export DATASET="cifar10"


python train_search_hfai.py --device=0 --num_obj=2 --num_states=1 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=1 --num_obj=2 --num_states=2 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=2 --num_obj=2 --num_states=3 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=3 --num_obj=2 --num_states=4 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=4 --num_obj=2 --num_states=5 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=5 --num_obj=2 --num_states=6 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=6 --num_obj=2 --num_states=7 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=7 --num_obj=2 --num_states=8 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline

# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10_resnet50_num_states=15
# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name noCEFormat_cifar100_resnet50_num_states=20
# hfai bash hfai_run_ceformat_different_size_state.sh -- -n 1 --force --no_diff --name experiments_size_of_state1-8
