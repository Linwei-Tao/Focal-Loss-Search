#!/bin/bash

export B1_MAX_LIST=("0.001" "0.002" "0.01" "0.02" "0.1" "0.2")
export B1_M_LIST=(1 2 3)
export ETA_LIST=(["0"]="ddim" ["1"]="ddpm") # ["0"]="ddim" ["1"]="ddpm"
export SAMPLE_TIMESTEPS_LIST=("100")

cd /home/xiyuwang/code/momentum-diffusion/runs/cifar10s

for ETA in "${!ETA_LIST[@]}"
  do
    for B1_M in "${B1_M_LIST[@]}"
      do
        for B1_MAX in "${B1_MAX_LIST[@]}"
          do
            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
              do
                hfai bash cifar10_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
                --name cifar10_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
              done
          done
      done
    hfai bash cifar10_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
    --name cifar10_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done
