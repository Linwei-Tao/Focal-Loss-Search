#!/bin/bash

N_STATES=(14 15 16 17 18 19 20)
MODELS=("resnet110" "wide_resnet" "densenet121" "resnet50")
DATASETS=("cifar10" "cifar100")

for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#              cmd="hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach"
#              echo ${cmd}
#              eval ${cmd}
#              sleep 60s
          done
      done
  done


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#              cmd="hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach"
#              echo ${cmd}
#              eval ${cmd}
#              sleep 60s
          done
      done
  done