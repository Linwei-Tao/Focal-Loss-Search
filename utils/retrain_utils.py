'''
Script for training models.
'''
import os
import socket
import time
import numpy as np
from torch import optim
import torch
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import random
import json
import sys
import wandb

# Import dataloaders
import dataset.cifar10 as cifar10
import dataset.cifar100 as cifar100
import dataset.tiny_imagenet as tiny_imagenet

from module.resnet import resnet50

# Import train and validation utilities
from utils.train_utils import model_train
from utils.valid_utils import model_valid
import utils

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}


def retrain(lossfunc, args, wandb):
    net = resnet50(num_classes=10)
    net = net.cuda()

    num_epochs = args.retrain_epochs

    optimizer = optim.SGD(net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250],
                                               gamma=0.1)
    if (args.dataset == 'tiny_imagenet'):
        train_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.data,
            split='train',
            batch_size=128,
            pin_memory=True,
            data_dir=args.data
        )

        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.data,
            split='val',
            batch_size=128,
            pin_memory=True,
            data_dir=args.data
        )

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.data,
            split='val',
            batch_size=128,
            pin_memory=True,
            data_dir=args.data
        )
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=128,
            augment=True,
            random_seed=1,
            pin_memory=True,
            data_dir=args.data
        )

        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=128,
            pin_memory=True,
            data_dir=args.data
        )

    for epoch in range(num_epochs):
        model_train(train_loader, net, lossfunc, optimizer, "Retrain Epoch: {}/{}".format(epoch + 1, num_epochs), args,
                    gumbel_training=False)

        # test on test set
        retrain_test_pre_accuracy, retrain_test_pre_ece, retrain_test_pre_adaece, retrain_test_pre_cece, \
        retrain_test_pre_nll, retrain_test_T_opt, retrain_test_post_ece, retrain_test_post_adaece, \
        retrain_test_post_cece, retrain_test_post_nll = model_valid(
            test_loader, val_loader, net)

        wandb.log({
            "retrain_test_pre_accuracy": retrain_test_pre_accuracy * 100,
            "retrain_test_pre_ece": retrain_test_pre_ece * 100,
            "retrain_test_pre_adaece": retrain_test_pre_adaece * 100,
            "retrain_test_pre_cece": retrain_test_pre_cece * 100,
            "retrain_test_pre_nll": retrain_test_pre_nll * 100, "retrain_test_T_opt": retrain_test_T_opt,
            "retrain_test_post_ece": retrain_test_post_ece * 100,
            "retrain_test_post_adaece": retrain_test_post_adaece * 100,
            "retrain_test_post_cece": retrain_test_post_cece * 100,
            "retrain_test_post_nll": retrain_test_post_nll * 100,
        })

        print("[Retrain Epoch: {}/{}] Test Accuracy: {}, Test ECE: {}".format(epoch + 1, num_epochs,
                                                                              retrain_test_pre_accuracy,
                                                                              retrain_test_pre_ece))
        utils.save(net, os.path.join(args.save, 'model-weights-retrain.pt'))

        scheduler.step()
