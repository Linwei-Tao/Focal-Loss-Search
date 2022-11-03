#!/bin/bash

N_STATES=(20 19 18 17 16 15 14)
MODELS=("wide_resnet" "densenet121" "resnet50" "resnet110" )
DATASETS=("cifar100" "cifar10")

for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#               hfai stop Search_Retrain_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"
          done
      done
  done


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#              hfai stop Search_Retrain_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"
          done
      done
  done