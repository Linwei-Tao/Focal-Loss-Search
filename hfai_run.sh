#!/bin/bash

export NSTATE=$1
export MODEL=$2
export DATASET=$3

python train_search_hfai.py --device=0 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=1 --lfs_lambda=1 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=1 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=2 --lfs_lambda=2 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=2 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=5 --lfs_lambda=5 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=3 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=10 --lfs_lambda=10 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=4 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=5 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=50 --lfs_lambda=50 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=6 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=7 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=200 --lfs_lambda=200 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline

# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10_resnet50_num_states=15
# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name noCEFormat_cifar100_resnet50_num_states=20
