import wandb


# resnet50 cifar100
wandb.init(
    project="Focal Loss Search Calibration",
    entity="linweitao",
    config={
        "model": "resnet50",
        "dataset": "cifar100",
    },
    tags=["baseline"],
    name="Cross Entropy"
)

wandb.log(
    {
        "retrain_test_pre_accuracy": 76.7,
        "retrain_test_pre_ece": 17.52,
        "retrain_test_pre_adaece": 17.52,
        "retrain_test_pre_cece": 0.38,
        "retrain_test_pre_nll": 153.67,
        "retrain_test_T_opt": 2.1,
        "retrain_test_post_ece": 3.42,
        "retrain_test_post_adaece": 3.42,
        "retrain_test_post_cece": 0.22,
        "retrain_test_post_nll": 106.83
    }
)