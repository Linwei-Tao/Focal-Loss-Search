#!/bin/bash

export DATASET="CIFAR10"
export METHOD="adam"
export SEED=96
export B1_MAX=$1

cd ../..

python image_generator.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED} --b1-max ${B1_MAX}

python main_fid.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED} --b1-max ${B1_MAX} -b 50