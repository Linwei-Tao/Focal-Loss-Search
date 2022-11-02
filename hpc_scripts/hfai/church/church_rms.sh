#!/bin/bash

export DATASET="CHURCH"
export METHOD="rms"
export SEED=96

cd ../..

python image_generator.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED}

python main_fid.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED} -b 50