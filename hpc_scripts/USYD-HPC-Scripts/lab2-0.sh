#!/bin/bash

cd /home/linwei/Desktop/projects/NAS-Calibration
for a in 'ResNet18' 'ResNet34' 'ResNet50' 'ResNet101' 'ResNet152' 'ShuffleNet' 'ShuffleNetV2' 'MobileNet' 'MobileNetV2' 'EfficientNet'
do
   CUDA_VISIBLE_DEVICES=0 python3 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$a --criterion=focal --focal_gamma=3 --grad_clip=2
done
