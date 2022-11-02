#!/bin/bash

export DATASET="CELEBA"
export METHOD="momentum"
export SEED=96
export B1_MAX=$1
export B1_M=$2
export ETA=$3
export SAMPLE_TIMESTEPS=$4

cd ../..

python image_generator.py -m "${METHOD}" --cfg configs/${DATASET}.yaml --seed "${SEED}" \
--b1-max "${B1_MAX}" --b1-m "${B1_M}" --eta "${ETA}" --st "${SAMPLE_TIMESTEPS}"

python main_fid.py -m "${METHOD}" --cfg configs/"${DATASET}".yaml --seed "${SEED}" -b 50 \
--b1-max "${B1_MAX}" --b1-m "${B1_M}" --eta "${ETA}" --st "${SAMPLE_TIMESTEPS}"