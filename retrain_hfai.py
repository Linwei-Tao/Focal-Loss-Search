import argparse
import wandb
import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from scipy.stats import kendalltau
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import glob

from module.loss_searcher import LFS
from module.memory import Memory
from module.predictor import Predictor
from module.loss import LossFunc
from module.loss_rejector import LossRejector

# Import utilities
from utils import gumbel_like, MO_MSE

from utils.predictor_utils import predictor_train
from utils.lfs_utils import search

# hfai
import hfai
import hfai_env

hfai_env.set_env('lfs')

# retrain
# Import dataloaders
import dataset.cifar10 as cifar10
import dataset.cifar100 as cifar100
import dataset.tiny_imagenet as tiny_imagenet

# Import network models
from module.resnet import resnet50, resnet110
from module.resnet_tiny_imagenet import resnet50 as resnet50_ti
from module.wide_resnet import wide_resnet_cifar
from module.densenet import densenet121

# Import train and validation utilities
from utils.train_utils import model_train
from utils.valid_utils import model_valid
import utils

dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet
}

models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def main():
    # set random seeds
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)  # set random seed: numpy
    torch.manual_seed(args.seed)  # set random seed: torch
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda

    print("args = %s", args)


    # --- load searched loss ---
    if args.load_checkpoints and (checkpoint_path / 'lossfunc_latest.pickle').exists():
        lossfunc = utils.pickle_load(os.path.join(args.load_checkpoints, 'lossfunc_latest.pickle'))
        wandb.config.update({
            "searched_loss_str": lossfunc.loss_str(no_gumbel=True),
            "num_states": lossfunc.num_states,
            "noCEFormat": lossfunc.noCEFormat
        }, allow_val_change=True)
        print(f"*********** Successfully continue lossfunc: {lossfunc.loss_str(no_gumbel=True)}.*********** ")
        print(lossfunc.alphas_ops)
    elif args.load_searched_loss:
        lossfunc = LossFunc(searched_loss=args.load_searched_loss)
        wandb.config.update({
            "searched_loss_str": lossfunc.loss_str(no_gumbel=True),
            "num_states": lossfunc.num_states,
            "noCEFormat": lossfunc.noCEFormat
        }, allow_val_change=True)
        print(f"*********** Successfully continue lossfunc: {lossfunc.loss_str(no_gumbel=True)}.*********** ")
    else:
        print(f"No loss found!")

    # --- retrain on searched loss ---
    if args.retrain_epochs > 0:
        # build model
        num_classes = dataset_num_classes[args.dataset]
        model = models[args.model](num_classes=num_classes).cuda()

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.1,
                                    momentum=0.9,
                                    weight_decay=5e-4,
                                    nesterov=False)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250],
                                                         gamma=0.1)

        if args.dataset == 'tiny_imagenet':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=0.1,
                                        momentum=0.9,
                                        weight_decay=5e-4,
                                        nesterov=False)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60],
                                                             gamma=0.1)

        if args.dataset == 'tiny_imagenet':
            train_loader = dataset_loader[args.dataset].get_data_loader(
                root="dataset/tiny-imagenet-200/",
                split='train',
                batch_size=64,
                pin_memory=True,
            )

            val_loader = dataset_loader[args.dataset].get_data_loader(
                root="dataset/tiny-imagenet-200/",
                split='val',
                batch_size=128,
                pin_memory=True,
            )

            test_loader = dataset_loader[args.dataset].get_data_loader(
                root="dataset/tiny-imagenet-200/",
                split='val',
                batch_size=128,
                pin_memory=True,
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

        start_epoch = 0
        if (args.save_path / 'retrain_latest.pt').exists():
            ckpt = torch.load(args.save_path / 'retrain_latest.pt', map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch']
            print(f"*********** Successfully continue retrain form epoch {start_epoch}.***********")

        for epoch in range(start_epoch, args.retrain_epochs):
            model_train(train_loader, model, lossfunc, optimizer,
                        "[{}] Retrain Epoch: {}/{}".format(args.device, epoch + 1, args.retrain_epochs),
                        args,
                        gumbel_training=False)

            # test on test set
            retrain_test_pre_accuracy, retrain_test_pre_ece, retrain_test_pre_adaece, retrain_test_pre_cece, \
            retrain_test_pre_nll, retrain_test_T_opt, retrain_test_post_ece, retrain_test_post_adaece, \
            retrain_test_post_cece, retrain_test_post_nll = model_valid(
                test_loader, val_loader, model)

            wandb.log({
                "retrain_test_pre_accuracy": retrain_test_pre_accuracy * 100,
                "retrain_test_pre_ece": retrain_test_pre_ece * 100,
                "retrain_test_pre_adaece": retrain_test_pre_adaece * 100,
                "retrain_test_pre_cece": retrain_test_pre_cece * 100,
                "retrain_test_pre_nll": retrain_test_pre_nll * 100,
                "retrain_test_T_opt": retrain_test_T_opt,
                "retrain_test_post_ece": retrain_test_post_ece * 100,
                "retrain_test_post_adaece": retrain_test_post_adaece * 100,
                "retrain_test_post_cece": retrain_test_post_cece * 100,
                "retrain_test_post_nll": retrain_test_post_nll * 100,
            })

            print(f"[[{args.device}] Retrain Epoch: {epoch + 1}/{args.retrain_epochs}]  "
                  f"retrain_test_pre_accuracy: {retrain_test_pre_accuracy * 100}  "
                  f"retrain_test_pre_ece: {retrain_test_pre_ece * 100}  "
                  f"retrain_test_pre_adaece: {retrain_test_pre_adaece * 100}  "
                  f"retrain_test_pre_cece: {retrain_test_pre_cece * 100}  "
                  f"retrain_test_pre_nll: {retrain_test_pre_nll * 100}  "
                  f"retrain_test_T_opt: {retrain_test_T_opt}  "
                  f"retrain_test_post_ece: {retrain_test_post_ece * 100}  "
                  f"retrain_test_post_adaece: {retrain_test_post_adaece * 100}  "
                  f"retrain_test_post_cece: {retrain_test_post_cece * 100}  "
                  f"retrain_test_post_nll: {retrain_test_post_nll * 100}  ")

            # store search checkpoint
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
            }
            torch.save(state, os.path.join(args.save, 'retrain_latest.pt'))
            scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Loss Function Search")
    # model
    parser.add_argument('--model', type=str, default='resnet50')

    # data
    parser.add_argument('--data', type=str, default='/data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10')

    # retrain
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    # load setting
    parser.add_argument('--load_checkpoints', type=str, default=None)
    parser.add_argument('--load_searched_loss', type=str, default=None)

    # others
    parser.add_argument('--seed', type=int, default=1, help='random seed')  # seed
    parser.add_argument('--device', type=int, default=0)  # gpu
    parser.add_argument('--wandb_mode', type=str, default="offline")  # wandb
    parser.add_argument('--platform', type=str, default='hfai')  # train_platform

    args, unknown_args = parser.parse_known_args()
    # for local run
    if args.platform == "local":
        os.environ["MARSV2_NB_NAME"] = str("12516")

    args.save = 'checkpoints/{}-{}'.format(os.environ["MARSV2_NB_NAME"], args.device)

    args.save_path = Path(args.save)
    args.save_path.mkdir(exist_ok=True, parents=True)
    if args.load_checkpoints:
        checkpoint_path = Path(args.load_checkpoints)

    # set current device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ['WANDB_MODE'] = args.wandb_mode

    # load dir
    args.retrain_epochs = 100 if args.dataset == 'tiny_imagenet' else 350

    wandb.login(key="960eed671fd0ffd9b830069eb2b49e77af2e73f2")
    args.wandb_dir = "./wandb_local" if args.platform == "local" else f"./wandb/{os.environ['MARSV2_NB_NAME']}"
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb.init(project="Focal Loss Sdearch Calibration", entity="linweitao", config=args,
               id="{}-{}".format(os.environ["MARSV2_NB_NAME"], args.device), dir=args.wandb_dir, resume="allow")

    print("wandb.run.dir", wandb.run.dir)
    main()
