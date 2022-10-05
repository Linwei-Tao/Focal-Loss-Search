import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

import utils
from genotypes import arch_fs0, arch_fs1, arch_fs2, arch_fs3, arch_fs4
from module.model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
# data
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# save
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
# training setting
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--train_batch', type=int, default=50, help='number of batches for quick train')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# model setting
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
# others
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

args.save = 'checkpoints/quick-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(
    path=args.save,
    scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # loss function
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')

    # load data
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    # generate data indices
    num_train = len(train_data)
    indices = list(range(num_train))
    split = args.train_batch * args.batch_size

    # build data loader
    train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers
    )
    valid_queue = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.num_workers
    )

    for genotype in [
        arch_fs0, arch_fs1, arch_fs2, arch_fs3, arch_fs4
        # arch_gae_0, arch_gae_1, arch_gae_2, PVLL_NAS, DARTS, PC_DARTS_cifar, NASNet
    ]:
        logging.info('genotype = %s' % (genotype,))
        # build model
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        model = model.to('cuda')
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        # build optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        # set lr scheduler
        scheduler = utils.LRScheduler(optimizer=optimizer,
                                      schedule='cyclic',
                                      total_epochs=float(args.epochs),
                                      lr_min=1e-5)

        # clear best acc recording
        best_acc = 0.
        # start training
        for epoch in range(args.epochs):
            logging.info('epoch %d lr %s', epoch, scheduler.get_lr())
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj = train(train_queue, model, criterion, optimizer)

            with torch.no_grad():
                valid_acc, valid_obj = infer(valid_queue, model, criterion)

            logging.info('epoch %03d overall train_acc=%.4f valid_acc=%.4f', epoch, train_acc, valid_acc)

            best_acc = max(valid_acc, best_acc)
            logging.info('[*] best accuracy: %.4f', best_acc)

            scheduler.step()

        del model, optimizer, scheduler


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (inputs, target) in enumerate(train_queue):
        # data to CUDA
        inputs = inputs.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)

        optimizer.zero_grad()
        logits, logits_aux = model(inputs)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (inputs, target) in enumerate(valid_queue):
        # data to CUDA
        inputs = inputs.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)

        logits, _ = model(inputs)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
