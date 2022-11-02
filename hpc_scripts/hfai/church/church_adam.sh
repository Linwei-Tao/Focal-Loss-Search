#!/bin/bash

export DATASET="CHURCH"
export METHOD="adam"
export SEED=96
export B1_MAX=$1
export B1_M=$2
export B2=$3
export ETA=$4
export SAMPLE_TIMESTEPS=$5

cd ../..

python image_generator.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED} \
--b1-max ${B1_MAX} --b2 ${B2} --b1-m ${B1_M} --eta ${ETA}

#python fake_img_lmdb.py -m ${METHOD} --dataset ${DATASET} --seed ${SEED} \
#--b1-max ${B1_MAX} --b2 ${B2}  --b1-m ${B1_M} --eta ${ETA}

python main_fid.py -m ${METHOD} --cfg configs/${DATASET}.yaml --seed ${SEED} \
--b1-max ${B1_MAX} -b 50 --b2 ${B2}  --b1-m ${B1_M} --eta ${ETA}