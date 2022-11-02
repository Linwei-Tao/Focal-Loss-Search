#!/bin/bash
export B1_MAXS=("0.002")
export B2S=("0.999" "0.995" "0.99" "0.9")
export B1_MS=(1)
export ETAS=(["1"]="ddpm") # ["0"]="ddim" ["1"]="ddpm"
export SAMPLE_TIMESTEPS="100"

for ETA in "${!ETAS[@]}"
  do
    for B1_M in "${B1_MS[@]}"
      do
        for B1_MAX in "${B1_MAXS[@]}"
          do
            for B2 in "${B2S[@]}"
              do
                hfai bash church_adam.sh ${B1_MAX} ${B1_M} ${B2} ${ETA} ${SAMPLE_TIMESTEPS} -- -n 1 -f --detach \
                --no_diff --name church_adam_b1_${B1_MAX}_${B1_M}_b2_${B2}_${ETAS[${ETA}]}_st_${SAMPLE_TIMESTEPS}
              done
          done
      done
  done