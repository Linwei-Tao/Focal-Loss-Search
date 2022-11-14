#!/bin/bash

export MODEL="resnet50"
export DATASET="cifar10"


python train_search_hfai.py --noCEFormat --device=0 --seed=0 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=50 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=1 --seed=1 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=50 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=2 --seed=2 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=200 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=3 --seed=3 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=100 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=4 --seed=4 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=100 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=5 --seed=5 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=50 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=6 --seed=6 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=200 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --noCEFormat --device=7 --seed=7 --num_obj=2 --num_states=20 --predictor_lambda=25 --lfs_lambda=25 --predictor_warm_up=2000 --warm_up_population=200 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline

# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name cifar10_resnet50_num_states=15
# hfai bash hfai_run.sh -- -n 1 --force --no_diff --name noCEFormat_cifar100_resnet50_num_states=20
# hfai bash hfai_run_ceformat_different_size_state.sh -- -n 1 --force --no_diff --name experiments_size_of_state6-13
# hfai bash hfai_run_noceformat_warmuppopulation50-200.sh -- -n 1 --force --no_diff --name experiments_noce_warmuppopulation
