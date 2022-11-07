#!/bin/bash

#export CKPT=$1
#
#python retrain.py --device=0 --data=hfai --model=resnet50 --dataset=cifar10 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=1 --data=hfai --model=resnet110 --dataset=cifar10 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=2 --data=hfai --model=wide_resnet --dataset=cifar10 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=3 --data=hfai --model=densenet121 --dataset=cifar10 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=4 --data=hfai --model=resnet50 --dataset=cifar100 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=5 --data=hfai --model=resnet110 --dataset=cifar100 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=6 --data=hfai --model=wide_resnet --dataset=cifar100 --wandb_mode=offline --load_checkpoints="${CKPT}" &
#python retrain.py --device=7 --data=hfai --model=densenet121 --dataset=cifar100 --wandb_mode=offline --load_checkpoints="${CKPT}"



NSTATE=$1
MODEL=$2
DATASET=$3
DEVICE=$4


python retrain_hfai.py --device=0 --data=hfai --model=--dataset=tiny_imagenet  --model=resnet50_ti --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=1 --data=hfai --model=resnet110 --dataset=cifar10 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=2 --data=hfai --model=wide_resnet --dataset=cifar10 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=3 --data=hfai --model=densenet121 --dataset=cifar10 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=4 --data=hfai --model=resnet50 --dataset=cifar100 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=5 --data=hfai --model=resnet110 --dataset=cifar100 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=6 --data=hfai --model=wide_resnet --dataset=cifar100 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &
python retrain_hfai.py --device=7 --data=hfai --model=densenet121 --dataset=cifar100 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" &



#hfai python retrain_hfai.py --device=0 --data=hfai --model=densenet121 --dataset=cifar100 --wandb_mode=offline --load_checkpoints=checkpoints/Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0 -- -n 1 -f  --no_diff