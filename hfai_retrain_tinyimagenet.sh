#!/bin/bash

python retrain.py --device=0 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=1 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=2 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=3 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=4 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=5 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=6 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}" &
python retrain.py --device=7 --data=hfai --model=resnet50_ti --dataset=tiny_imagenet --wandb_mode=offline --load_checkpoints="${CKPT}"

# hfai bash hfai_retrain_tinyimagenet.sh -- -n 1 --force --no_diff --name Retrain_TI11041114