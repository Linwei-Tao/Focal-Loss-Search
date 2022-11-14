#!/bin/bash

export MODEL="resnet50"
export DATASET="cifar10"


python train_search_hfai.py --device=0 --num_obj=2 --noCEFormat --num_states=9 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=1 --num_obj=2 --noCEFormat --num_states=10 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=2 --num_obj=2 --noCEFormat --num_states=11 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=3 --num_obj=2 --noCEFormat --num_states=12 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=4 --num_obj=2 --noCEFormat --num_states=13 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=5 --num_obj=2 --noCEFormat --num_states=14 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=6 --num_obj=2 --noCEFormat --num_states=15 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=7 --num_obj=2 --noCEFormat --num_states=16 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline

# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10_resnet50_num_states=15
# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name noCEFormat_cifar100_resnet50_num_states=20
# hfai bash hfai_run_noceformat_different_size_state9-16.sh -- -n 1 --force --no_diff
