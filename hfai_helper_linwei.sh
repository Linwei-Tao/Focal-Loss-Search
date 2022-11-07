## download files with maximum timeout
## hfai workspace download wandb/ -s 21600 -o 21600 -t 21600 -l 7200 -f -n
#
#
## download and sync data
FILES=(
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
"wandb/wandb/offline-run-20221105_174245-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221105_204911-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221105_225219-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221106_012534-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221106_012814-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-1"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-4"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-3"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-2"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-0"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-6"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-7"
"wandb/wandb/offline-run-20221106_013555-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=20-5"
)

cd /mnt/LFS
for FILE in "${FILES[@]}"
  do
    hfai workspace download "${FILE}"
    wandb sync "${FILE}"
#    rm -rf "${FILE}"
  done

#cd /mnt/LFS
#for FILE in "${FILES[@]}"
#  do
#    wandb sync "${FILE}"
#  done


#cd /mnt/LFS/wandb
#wandb sync --sync-all

#for FILE in "${FILES[@]}"
#  do
#    rm -rf "${FILE}"
#  done
#



