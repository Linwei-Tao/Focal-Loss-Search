#!/bin/bash

export NSTATE="20"
export MODEL="resnet50"
export DATASET="cifar10"

python train_search_hfai.py --device=0 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=50 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=1 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=100 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=2 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=200 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=3 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=300 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=4 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=400 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=5 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=500 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=6 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=800 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline &
python train_search_hfai.py --device=7 --noCEFormat --num_obj=2 --num_states="${NSTATE}" --predictor_lambda=100 --lfs_lambda=100 --warm_up_population=1000 --predictor_warm_up=2000 --data=hfai --model="${MODEL}" --dataset="${DATASET}" --wandb_mode=offline

# hfai bash hfai_warmup_population.sh -- -n 1 --force --no_diff --name Experiment_WarmupPopulation