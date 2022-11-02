#!/bin/bash

export DATASET="CHURCH"
export METHOD="o"
export SEED=96
export ETA=$1
export SAMPLE_TIMESTEPS=$2

cd ../..

python image_generator.py -m "${METHOD}" --cfg configs/${DATASET}.yaml --seed "${SEED}" \
--eta "${ETA}" --st "${SAMPLE_TIMESTEPS}"

python main_fid.py -m "${METHOD}" --cfg configs/"${DATASET}".yaml --seed "${SEED}" -b 50 \
--eta "${ETA}" --st "${SAMPLE_TIMESTEPS}"