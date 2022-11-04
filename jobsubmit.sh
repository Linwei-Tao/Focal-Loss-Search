#!/bin/bash

################################################
#      search train for cifar10 & cifar100     #
################################################

N_STATES=(20 19 18 17 16 15 14)
MODELS=("resnet50" "wide_resnet" "densenet121" "resnet110" )
DATASETS=("cifar10" "cifar100")

for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11041114_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
          done
      done
  done


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11041114_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
          done
      done
  done







#################################################
##      search train for tiny-imagenet          #
#################################################

N_STATES=(20 19 18 17 16 15 14)
MODELS=("resnet50_ti" )
DATASETS=("tiny_imagenet")


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11041114_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
          done
      done
  done


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for NSTATE in "${N_STATES[@]}"
          do
              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11041114_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
          done
      done
  done






##################################################
###      retrain on cifar10 and cifar100         #
##################################################
#
#CKPTS=(20 19 18 17 16 15 14)
#
#
#for CKPT in "${CKPTS[@]}"
#  do
#       hfai bash hfai_run.sh "${CKPT}" -- -n 1 --force --no_diff --name Retrain_CIFAR11041114_CPKT="${CPKT}" --detach
#  done
