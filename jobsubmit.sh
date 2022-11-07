#!/bin/bash

#################################################
##      search train for cifar10 & cifar100     #
#################################################
#
#N_STATES=(20 19 18 17 16 15 14)
#MODELS=("resnet50" "wide_resnet" "densenet121" "resnet110" )
#DATASETS=("cifar10" "cifar100")
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for NSTATE in "${N_STATES[@]}"
#          do
#               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11050540_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#          done
#      done
#  done
#
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for NSTATE in "${N_STATES[@]}"
#          do
#              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#          done
#      done
#  done
#
#
#
#
#
#
#
##################################################
##      search train for tiny-imagenet           #
##################################################
#
#N_STATES=(20 19 18 17 16 15 14)
#MODELS=("resnet50_ti" )
#DATASETS=("tiny_imagenet")
#
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for NSTATE in "${N_STATES[@]}"
#          do
#               hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11050540_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#          done
#      done
#  done
#
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for NSTATE in "${N_STATES[@]}"
#          do
#              hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" -- -n 1 --force --no_diff --name Search_Retrain11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}" --detach
#          done
#      done
#  done






##################################################
###      retrain on cifar10 and cifar100         #
##################################################
#
#CKPTS=()
#
#
#for CKPT in "${CKPTS[@]}"
#  do
#       hfai bash hfai_run.sh "${CKPT}" -- -n 1 --force --no_diff --name Retrain_CIFAR11050540_CPKT="${CPKT}" --detach
#  done




#hfai bash hfai_retrain_searched_loss.sh -- -n 1 --force --no_diff --name Retrain_SearchedLoss_GradClip --detach




##################################################
###      retrain on multiple cases               #
##################################################
##N_STATES=(20 19 18 17 16 15 14)
##MODELS=("resnet50" "wide_resnet" "densenet121" "resnet110" )
##DATASETS=("cifar100" "cifar10")
#
#N_STATES=(20 19 18 17 16 15 14)
#MODELS=("resnet50")
#DATASETS=("cifar10")
#DEVICES=(0 1 2 3 4 5 6 7)
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for DEVICE in "${DEVICES[@]}"
#          do
#            for NSTATE in "${N_STATES[@]}"
#              do
#                   hfai bash hfai_retrain_mutiple_cases.sh "${NSTATE}" "${MODEL}" "${DATASET}" "${DEVICE}" -- -n 1 --force --no_diff --name RetrainMultiCase11050540_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" --detach
#              done
#          done
#      done
#  done
#
#
#for DATASET in "${DATASETS[@]}"
#  do
#    for MODEL in "${MODELS[@]}"
#      do
#        for DEVICE in "${DEVICES[@]}"
#          do
#            for NSTATE in "${N_STATES[@]}"
#              do
#                   hfai bash hfai_retrain_mutiple_cases_CE.sh "${NSTATE}" "${MODEL}" "${DATASET}" "${DEVICE}" -- -n 1 --force --no_diff --name RetrainMultiCase11050540_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${DEVICE}" --detach
#              done
#          done
#      done
#  done




#################################################
##      train wideresnet                        #
#################################################
N_STATES=(20 19 18 17 16 15 14)
MODELS=("wide_resnet")
DATASETS=("cifar100")
SEARCH_EPOCHS=(300 500)


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for SEARCH_EPOCH in "${SEARCH_EPOCHS[@]}"
          do
            for NSTATE in "${N_STATES[@]}"
              do
                hfai bash hfai_run.sh "${NSTATE}" "${MODEL}" "${DATASET}" "${SEARCH_EPOCH}" -- -n 1 --force --no_diff --name Wideresnet_more_search_noCEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${SEARCH_EPOCH}" --detach
              done
          done
      done
  done


for DATASET in "${DATASETS[@]}"
  do
    for MODEL in "${MODELS[@]}"
      do
        for SEARCH_EPOCH in "${SEARCH_EPOCHS[@]}"
          do
            for NSTATE in "${N_STATES[@]}"
              do
                hfai bash hfai_run_ceformat.sh "${NSTATE}" "${MODEL}" "${DATASET}" "${SEARCH_EPOCH}" -- -n 1 --force --no_diff --name Wideresnet_more_search_CEFormat_"${DATASET}"_"${MODEL}"_num_states="${NSTATE}"-"${SEARCH_EPOCH}" --detach
              done
          done
      done
  done