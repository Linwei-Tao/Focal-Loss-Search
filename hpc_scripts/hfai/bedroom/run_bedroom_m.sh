#!/bin/bash

export B1_MAX_LIST=("0.01" "0.03" "0.05" "0.09" "0.1")
export B1_M_LIST=(2)
export ETA_LIST=(["1"]="ddpm")
#export SAMPLE_TIMESTEPS_LIST=("50")
export SAMPLE_TIMESTEPS=50

cd /home/xiyuwang/code/momentum-diffusion/runs/bedroom

for ETA in "${!ETA_LIST[@]}"
  do
#    for B1_M in "${B1_M_LIST[@]}"
#      do
#        for B1_MAX in "${B1_MAX_LIST[@]}"
#          do
#            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
#              do
#                hfai bash bedroom_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
#                --name bedroom_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
#              done
#          done
#      done
        hfai bash bedroom_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name bedroom_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done

export B1_MAX_LIST=("0.001" "0.0015" "0.002" "0.0025")
export ETA_LIST=(["0"]="ddim")

for ETA in "${!ETA_LIST[@]}"
  do
#    for B1_M in "${B1_M_LIST[@]}"
#      do
#        for B1_MAX in "${B1_MAX_LIST[@]}"
#          do
#            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
#              do
#                hfai bash bedroom_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
#                --name bedroom_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
#              done
#          done
#      done
        hfai bash bedroom_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name bedroom_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done
