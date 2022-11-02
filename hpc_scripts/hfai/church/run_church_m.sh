#!/bin/bash

export B1_MAX_LIST=("0.01" "0.015" "0.03" "0.05" "0.07" "0.09")
export B1_M_LIST=(2)
export ETA_LIST=(["1"]="ddpm") # ["0"]="ddim" ["1"]="ddpm"
export SAMPLE_TIMESTEPS_LIST=("1000")

cd /home/xiyuwang/code/momentum-diffusion/runs/church

for ETA in "${!ETA_LIST[@]}"
  do
    for B1_M in "${B1_M_LIST[@]}"
      do
        for B1_MAX in "${B1_MAX_LIST[@]}"
          do
            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
              do
                hfai bash church_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
                --name church_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
              done
          done
      done
        hfai bash church_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name church_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done

export B1_MAX_LIST=("0.001" "0.002" "0.003" "0.004")
export ETA_LIST=(["0"]="ddim") # ["0"]="ddim" ["1"]="ddpm"

for ETA in "${!ETA_LIST[@]}"
  do
    for B1_M in "${B1_M_LIST[@]}"
      do
        for B1_MAX in "${B1_MAX_LIST[@]}"
          do
            for SAMPLE_TIMESTEPS in "${SAMPLE_TIMESTEPS_LIST[@]}"
              do
                hfai bash church_m.sh "${B1_MAX}" "${B1_M}" "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
                --name church_m_b1_"${B1_MAX}"_"${B1_M}"_st_"${SAMPLE_TIMESTEPS}"_"${ETA_LIST[${ETA}]}" --detach
              done
          done
      done
        hfai bash church_o.sh "${ETA}" "${SAMPLE_TIMESTEPS}" -- -n 1 -f --no_diff \
        --name church_o_st_"${SAMPLE_TIMESTEPS}"_${ETA_LIST[${ETA}]} --detach
  done
