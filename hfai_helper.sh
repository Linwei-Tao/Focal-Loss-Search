# download files with maximum timeout
# hfai workspace download wandb/ -s 21600 -o 21600 -t 21600 -l 7200 -f -n


# download and sync data
FILES=("wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-4" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-1" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-7" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-2" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-6" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-0" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-3" "wandb/wandb/offline-run-20221105_174249-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-5" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-4" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-3" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-1" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-6" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-5" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-2" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-0" "wandb/wandb/offline-run-20221105_204923-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-7" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-7" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-1" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-4" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-0" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-6" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-2" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-3" "wandb/wandb/offline-run-20221106_014936-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-5" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-6" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-1" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-2" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-4" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-3" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-7" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-0" "wandb/wandb/offline-run-20221106_021110-Search_Retrain11050540_noCEFormat_cifar100_densenet121_num_states=16-5")
for FILE in "${FILES[@]}"
  do
    hfai workspace download "${FILE}"
  done

for FILE in "${FILES[@]}"
  do
    wandb sync "${FILE}"
  done

for FILE in "${FILES[@]}"
  do
    rm -rf "${FILE}"
  done




