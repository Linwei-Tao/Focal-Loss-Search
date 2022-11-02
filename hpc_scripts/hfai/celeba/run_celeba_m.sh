#!/bin/bash

export B1_MAX_LIST=("0.01" "0.03" "0.05" "0.1" "0.15" "0.2")
export B1_M_LIST=(2)
export ETA_LIST=(["1"]="ddpm")
export SAMPLE_TIMESTEPS_LIST=("50")

cd /home/xiyuwang/code/momentum-diffusion/runs/celeba

for ETA in "${!ETA_LIST[@]}"
  do
    for B1_M in "${B1_M_LIST[@]}"
      do
        for B1_MAX in "${B1_MAX_LIST[@]}"
          do
            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
              do
                hfai bash celeba_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
                --name celeba_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
              done
          done
      done
        hfai bash celeba_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name celeba_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done

export B1_MAX_LIST=("0.0005" "0.001" "0.0015" "0.002")
export ETA_LIST=(["0"]="ddim")

for ETA in "${!ETA_LIST[@]}"
  do
    for B1_M in "${B1_M_LIST[@]}"
      do
        for B1_MAX in "${B1_MAX_LIST[@]}"
          do
            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
              do
                hfai bash celeba_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
                --name celeba_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
              done
          done
      done
        hfai bash celeba_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name celeba_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done
