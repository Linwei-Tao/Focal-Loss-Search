#!/bin/bash

cd /home/linwei/Desktop/projects/NAS-Calibration
for a in 'ResNeXt29_32x4d' 'ResNet18' 'ResNet34'
do
   CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3 --grad_clip=2
done

# tmux -c 'bash /home/linwei/Desktop/projects/NAS-Calibration/exp_scripts/lab1-0.sh'