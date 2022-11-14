# download files with maximum timeout
# hfai workspace download wandb/ -s 21600 -o 21600 -t 21600 -l 7200 -f -n

# 3741867+ 3741866+ 3741851+ 3741850+ 3741823+ 3741822+ 3741821+ 3741819+ 3741818+ 3741817+ 3741803+ 3741794+ 3741775+
# 3741764+ 3741753+ 3741752+ 3741728+ 3741727+ 3741521+ 3741514+ 3741511+ 3741509+ 3741499+ 3741447+ 3741426+ 3741338+
# 3740652+ 3740556+


# download and sync data
FILES=(
)

cd /mnt/LFS
for FILE in "${FILES[@]}"
  do
    wandb sync "${FILE}"
    rm -rf "${FILE}"
  done



#DEVICES=(0 1 2 3 4 5 6 7)
#for DEVICE in "${DEVICES[@]}"
#  do
#    wandb sync --sync-all
#  done






















































































































