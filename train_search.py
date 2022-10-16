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
from scipy.stats import kendalltau
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

import utils
from module.loss_searcher import LFS
from module.memory import Memory
from module.predictor import Predictor
from utils import gumbel_like

from module.resnet import resnet50
from module.loss import LossFunc
from module.loss_rejector import LossRejector

# Import metrics to compute
from Metrics.metrics import test_classification_net_logits
from Metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import temperature scaling and NLL utilities
from temperature_scaling import ModelWithTemperature



CIFAR_CLASSES = 10



def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # enable GPU and set random seeds
    np.random.seed(args.seed)  # set random seed: numpy
    torch.cuda.set_device(args.gpu)

    # fast search
    cudnn.deterministic = False
    cudnn.benchmark = True

    torch.manual_seed(args.seed)  # set random seed: torch
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    if len(unknown_args) > 0:
        logging.warning('unknown_args: %s' % unknown_args)
    else:
        logging.info('unknown_args: %s' % unknown_args)
    # Loss Function Search

    # build the model with model_search.Network
    logging.info("init arch param")
    model = resnet50(num_classes=CIFAR_CLASSES)
    model = model.to('cuda')
    logging.info("model param size = %fMB", utils.count_parameters_in_MB(model))

    # use SGD to optimize the model (optimize model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # construct data transformer (including normalization, augmentation)
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    # load cifar10 data training set (train=True)
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # generate data indices
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # split training set and validation queue given indices
    # train queue:
    train_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers
    )

    # validation queue:
    valid_queue = DataLoader(
        train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.num_workers
    )

    # learning rate scheduler (with cosine annealing)
    scheduler = CosineAnnealingLR(optimizer, int(args.epochs), eta_min=args.learning_rate_min)

    # loss function
    lossfunc = LossFunc()


    # -- build model --
    predictor = Predictor(input_size=lossfunc.num_states + lossfunc.num_operator_choice,
                          hidden_size=args.predictor_hidden_state)
    predictor = predictor.to('cuda')

    logging.info("predictor param size = %fMB", utils.count_parameters_in_MB(predictor))

    logging.info('using MSE loss for predictor')
    predictor_criterion = F.mse_loss

    # loss function searcher
    lfs = LFS(
        lossfunc=lossfunc, model=model, momentum=args.momentum, weight_decay=args.weight_decay,
        lfs_learning_rate=args.lfs_learning_rate, lfs_weight_decay=args.lfs_weight_decay,
        predictor=predictor, pred_learning_rate=args.pred_learning_rate,
        lfs_criterion=F.mse_loss, predictor_criterion=predictor_criterion
    )

    loss_rejector = LossRejector(lossfunc, train_queue, model, num_rejection_sample=5)

    memory = Memory(limit=args.memory_size, batch_size=args.predictor_batch_size)

    # --- Part 1: model warm-up and build memory---
    # 1.1 model warm-up
    if args.load_model is not None:
        # load from file
        logging.info('Load warm-up from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model-weights-warm-up.pt')))
        warm_up_gumbel = utils.pickle_load(os.path.join(args.load_model, 'gumbel-warm-up.pickle'))
    else:
        # 1.1.1 sample cells for warm-up
        warm_up_gumbel = []
        # assert args.warm_up_population >= args.predictor_batch_size
        while len(warm_up_gumbel) < args.warm_up_population:
            g_ops = gumbel_like(lossfunc.alphas_ops)
            g_operators = gumbel_like(lossfunc.alphas_operators)
            flag, g_ops, g_operators = loss_rejector.evaluate_loss(g_ops, g_operators)
            if flag: warm_up_gumbel.append((g_ops, g_operators))
        utils.pickle_save(warm_up_gumbel, os.path.join(args.save, 'gumbel-warm-up.pickle'))
        # 1.1.2 warm up
        for epoch, gumbel in enumerate(warm_up_gumbel):
            logging.info('[warm-up model] epoch %d/%d', epoch + 1, args.warm_up_population)
            # warm-up
            lossfunc.g_ops, lossfunc.g_operators = gumbel
            print("Objection function: ",lossfunc.loss_str())
            objs, top1, top5 = model_train(train_queue, model, lossfunc, optimizer, name='warm-up model')
            logging.info('[warm-up model] epoch %d/%d overall loss=%.4f top1-acc=%.4f top5-acc=%.4f',
                         epoch + 1, args.warm_up_population, objs, top1, top5)
            # save weights
            utils.save(model, os.path.join(args.save, 'model-weights-warm-up.pt'))


    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        logging.info('Load valid model from %s', args.load_model)
        model.load_state_dict(torch.load(os.path.join(args.load_memory, 'model-weights-valid.pt')))
        memory.load_state_dict(
            utils.pickle_load(
                os.path.join(args.load_memory, 'memory-warm-up.pickle')
            )
        )
    else:
        for epoch, gumbel in enumerate(warm_up_gumbel):
            # re-sample Gumbel distribution
            lossfunc.g_ops, lossfunc.g_operators = gumbel
            # train model for one step
            objs, top1, top5 = model_train(train_queue, model, lossfunc, optimizer, name='build memory')
            logging.info('[build memory] train model-%03d loss=%.4f top1-acc=%.4f',
                         epoch + 1, objs, top1)
            # valid model
            p_accuracy, p_ece, p_adaece, p_cece, p_nll, T_opt, ece, adaece, cece, nll = model_valid(valid_queue, model)
            logging.info('[build memory] valid model-%03d nll=%.4f top1-acc=%.4f ece=%.4f',
                         epoch + 1, p_nll, p_accuracy, p_ece)
            # save to memory
            memory.append(weights=torch.stack([w.detach() for w in lossfunc.arch_weights()]),
                              nll=torch.tensor(nll, dtype=torch.float32).to('cuda'),
                              acc=torch.tensor(p_accuracy, dtype=torch.float32).to('cuda'),
                              ece=torch.tensor(p_ece, dtype=torch.float32).to('cuda'))
            # checkpoint: model, memory
            utils.save(model, os.path.join(args.save, 'model-weights-valid.pt'))
            utils.pickle_save(memory.state_dict(),
                              os.path.join(args.save, 'memory-warm-up.pickle'))

    logging.info('memory size=%d', len(memory))

    # --- Part 2 predictor warm-up ---
    if args.load_extractor is not None:
        logging.info('Load extractor from %s', args.load_extractor)
        lfs.predictor.extractor.load_state_dict(torch.load(args.load_extractor)['weights'])

    predictor.train()
    for epoch in range(args.predictor_warm_up):
        epoch += 1
        # warm-up
        p_loss, p_true, p_pred = predictor_train(lossfunc, memory)
        if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
            logging.info('[warm-up predictor] epoch %d/%d loss=%.4f', epoch, args.predictor_warm_up, p_loss)
            logging.info('\np-true: %s\np-pred: %s', p_true.data, p_pred.data)
            k_tau = kendalltau(p_true.detach().to('cpu'), p_pred.detach().to('cpu'))[0]
            logging.info('kendall\'s-tau=%.4f' % k_tau)
            # save predictor
            utils.save(lfs.predictor, os.path.join(args.save, 'predictor-warm-up.pt'))

    # --- Part 3 loss function search ---
    for epoch in range(args.epochs):
        # search
        objs, top1, top5, objp = search(train_queue, valid_queue, model, lfs, lossfunc, optimizer, memory)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))
        # update learning rate
        scheduler.step()
        # get current learning rate
        lr = scheduler.get_lr()[0]
        logging.info('[loss function search] epoch %d/%d lr %e', epoch + 1, args.epochs, lr)
        # log
        logging.info('[loss function search] overall loss=%.4f top1-acc=%.4f top5-acc=%.4f predictor_loss=%.4f',
                     objs, top1, top5, objp)

def model_train(train_queue, model, lossfunc, optimizer, name):
    # set model to training model
    model.train()
    # create metrics
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    # training loop
    total_steps = len(train_queue)
    for step, (x, target) in enumerate(train_queue):
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        # update model weight
        # forward
        optimizer.zero_grad()
        logits = model(x)
        loss = lossfunc(logits, target)
        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            logging.info('[%s] train model %03d/%03d loss=%.4f top1-acc=%.4f top5-acc=%.4f',
                         name, step, total_steps, objs.avg, top1.avg, top5.avg)
    # return average metrics
    return objs.avg, top1.avg, top5.avg


def model_valid(valid_queue, model):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    logits, labels = get_logits_labels(valid_queue, model)
    conf_matrix, p_accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    p_ece = ece_criterion(logits, labels).item()
    p_adaece = adaece_criterion(logits, labels).item()
    p_cece = cece_criterion(logits, labels).item()
    p_nll = nll_criterion(logits, labels).item()

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(valid_queue, cross_validate="ece")
    T_opt = scaled_model.get_temperature()
    logits, labels = get_logits_labels(valid_queue, scaled_model)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)

    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()



    return p_accuracy, p_ece, p_adaece, p_cece, p_nll, T_opt, ece, adaece, cece, nll

def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels

def predictor_train(lfs, memory, unsupervised=False):
    objs = utils.AverageMeter()
    batch = memory.get_batch()
    all_nll = []
    all_p = []
    for weights, nll, acc, ece in batch:
        n = nll.size(0)
        pred, loss = lfs.predictor_step(weights, nll)
        objs.update(loss.data.item(), n)
        all_nll.append(nll)
        all_p.append(pred)
    return objs.avg, torch.cat(all_nll), torch.cat(all_p)


def search(train_queue, valid_queue, model, lfs, lossfunc, optimizer, memory):
    # -- train model --
    gsw_normal, gsw_reduce = 1., 1.  # gumbel sampling weight
    lossfunc.g_operation = gumbel_like(model.alphas_operation) * gsw_normal
    lossfunc.g_operator = gumbel_like(model.alphas_reduce) * gsw_reduce
    # train model for one step
    model_train(train_queue, model, lossfunc, optimizer, name='build memory')
    # -- valid model --
    p_accuracy, p_ece, p_adaece, p_cece, p_nll, T_opt, ece, adaece, cece, nll = model_valid(valid_queue, model)
    # save validation to memory
    logging.info('[loss function search] append memory nll=%.4f top1-acc=%.4f ece=%.4f', p_nll, p_accuracy, p_ece)
    # save to memory
    memory.append(weights=torch.stack([w.detach() for w in lossfunc.arch_weights()]),
                  nll=torch.tensor(nll, dtype=torch.float32).to('cuda'),
                  acc=torch.tensor(p_accuracy, dtype=torch.float32).to('cuda'),
                  ece=torch.tensor(p_ece, dtype=torch.float32).to('cuda'))
    utils.pickle_save(memory.state_dict(),
                      os.path.join(args.save, 'memory-search.pickle'))

    # -- predictor train --
    lfs.predictor.train()
    # use memory to train predictor
    p_loss, p_true, p_pred = None, None, None
    k_tau = -float('inf')
    for _ in range(args.predictor_warm_up):
        p_loss, p_true, p_pred = predictor_train(lfs, memory)
        k_tau = kendalltau(p_true.detach().to('cpu'), p_pred.detach().to('cpu'))[0]
        if k_tau > 0.95: break
    logging.info('[loss function search] train predictor p_loss=%.4f\np-true: %s\np-pred: %s',
                 p_loss, p_true.data, p_pred.data)
    logging.info('kendall\'s-tau=%.4f' % k_tau)

    lfs.step()
    # log
    logging.info('[loss function search] update architecture')

    return nll, p_accuracy, p_ece, p_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    # data
    parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    # save
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    # training setting
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    # search setting
    parser.add_argument('--lfs_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--lfs_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    # predictor setting
    parser.add_argument('--predictor_type', type=str, default='lstm')
    parser.add_argument('--predictor_warm_up', type=int, default=500, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--load_extractor', type=str, default=None, help='load memory from file')
    # model setting
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # others
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--debug', action='store_true', default=False, help='set logging level to debug')

    # loss function search
    parser.add_argument('--operator_size', type=int, default=8)

    args, unknown_args = parser.parse_known_args()

    args.save = 'checkpoints/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(
        path=args.save,
        scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
    )

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging_level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=logging_level,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main()
