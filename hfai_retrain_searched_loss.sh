#!/bin/bash

LOSS=$1

python retrain_hfai.py --device=0 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar10  --model=resnet110  &
python retrain_hfai.py --device=1 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar10  --model=wide_resnet  &
python retrain_hfai.py --device=2 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar10  --model=densenet121  &
python retrain_hfai.py --device=3 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar100  --model=resnet50  &
python retrain_hfai.py --device=4 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar100  --model=resnet110 &
python retrain_hfai.py --device=5 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar100  --model=wide_resnet &
python retrain_hfai.py --device=6 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=cifar100  --model=densenet121 &
python retrain_hfai.py --device=7 --data=hfai --wandb_mode=offline --load_searched_loss="${LOSS}"  --dataset=tiny_imagenet  --model=resnet50_ti

# hfai python retrain_hfai.py --device=7 --data=hfai --wandb_mode=disabled --load_searched_loss=northern-deluge-377 --grad_clip=5 -- -n 1 -f
