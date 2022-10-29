#!/bin/bash

cd /home/linwei/Desktop/projects/NAS-Calibration
for a in 'ResNeXt29_2x64d' 'ResNeXt29_4x64d' 'ResNeXt29_8x64d' 'ResNeXt29_32x4d'
do
   CUDA_VISIBLE_DEVICES=1 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3 --grad_clip=2
done
