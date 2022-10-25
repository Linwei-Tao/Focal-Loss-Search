import argparse
import glob
import wandb
import os
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
from utils import gumbel_like, MO_MSE

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
    # set random seeds
    cudnn.deterministic = False
    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)  # set random seed: numpy
    torch.manual_seed(args.seed)  # set random seed: torch
    torch.cuda.manual_seed(args.seed)  # set random seed: torch.cuda

    print("args = %s", args)

    # build model
    model = resnet50(num_classes=CIFAR_CLASSES)
    model = model.cuda()
    wandb.config.model_size = utils.count_parameters_in_MB(model)

    # use SGD to optimize the model
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
    lossfunc = LossFunc(num_states=args.num_states, tau=args.tau)

    # -- build predictor --
    feature_num = args.num_states + 2  # state number + p_k and p_j
    predictor = Predictor(input_size=feature_num,
                          hidden_size=args.predictor_hidden_state,
                          num_obj=args.num_obj,
                          predictor_lambda=args.predictor_lambda).cuda()

    predictor_criterion = MO_MSE(args.lfs_lambda) if args.num_obj > 1 else F.mse_loss
    lfs_criterion = MO_MSE(args.lfs_lambda) if args.num_obj > 1 else F.mse_loss

    # loss function searcher
    lfs = LFS(
        lossfunc=lossfunc, model=model, momentum=args.momentum, weight_decay=args.weight_decay,
        lfs_learning_rate=args.lfs_learning_rate, lfs_weight_decay=args.lfs_weight_decay,
        predictor=predictor, pred_learning_rate=args.pred_learning_rate,
        lfs_criterion=lfs_criterion, predictor_criterion=predictor_criterion
    )

    # a loss evaluator that filter unpromissing loss function
    loss_rejector = LossRejector(lossfunc, train_queue, model, num_rejection_sample=5,
                                 threshold=args.loss_rejector_threshold)

    # a deque memory that store loss - (acc, ece, nll) pair for predictor training
    memory = Memory(limit=args.memory_size, batch_size=args.predictor_batch_size)

    # --- Part 1: model warm-up and build memory---
    # 1.1 model warm-up
    if args.load_model is not None:
        # load from file
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model-weights-warm-up.pt')))
        warm_up_gumbel = utils.pickle_load(os.path.join(args.load_model, 'gumbel-warm-up.pickle'))
    else:
        # 1.1.1 sample cells for warm-up
        warm_up_gumbel = []
        # assert args.warm_up_population >= args.predictor_batch_size
        while len(warm_up_gumbel) < args.warm_up_population:
            g_ops = gumbel_like(lossfunc.alphas_ops)
            flag, g_ops = loss_rejector.evaluate_loss(g_ops)
            if flag: warm_up_gumbel.append((g_ops))
        utils.pickle_save(warm_up_gumbel, os.path.join(args.save, 'gumbel-warm-up.pickle'))
        # 1.1.2 warm up
        for epoch, gumbel in enumerate(warm_up_gumbel):
            print('[warm-up model] epoch %d/%d' % (epoch + 1, args.warm_up_population))
            # warm-up
            lossfunc.g_ops = gumbel
            print("Objective function: %s" % (lossfunc.loss_str()))
            objs, top1, top5, nll = model_train(train_queue, model, lossfunc, optimizer, name='warm-up model')
            print('[warm-up model] epoch %d/%d searched loss=%.4f top1-acc=%.4f nll=%.4f' % (
                epoch + 1, args.warm_up_population, objs, top1, nll))
            # save weights
            utils.save(model, os.path.join(args.save, 'model-weights-warm-up.pt'))

    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        print('Load valid model from %s' % (args.load_model))
        model.load_state_dict(torch.load(os.path.join(args.load_memory, 'model-weights-valid.pt')))
        memory.load_state_dict(
            utils.pickle_load(
                os.path.join(args.load_memory, 'memory-warm-up.pickle')
            )
        )
    else:
        for epoch, gumbel in enumerate(warm_up_gumbel):
            # re-sample Gumbel distribution
            lossfunc.g_ops = gumbel
            # log function
            print("Objective function: %s" % (lossfunc.loss_str()))
            # train model for one step
            objs, top1, top5, nll = model_train(train_queue, model, lossfunc, optimizer, name='build memory')
            print('[build memory] train model-%03d searched_loss=%.4f train_top1-acc=%.4f nll=%.4f' % (
                epoch + 1, objs, top1, nll))
            # valid model
            pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll = model_valid(
                valid_queue, model)
            print('[build memory] valid model-%03d valid_nll=%.4f valid_top1-acc=%.4f valid_ece=%.4f' % (
                epoch + 1, pre_nll, pre_accuracy, pre_ece))
            # save to memory
            memory.append(weights=lossfunc.arch_weights(),
                          nll=torch.tensor(pre_nll, dtype=torch.float32).to('cuda'),
                          acc=torch.tensor(pre_accuracy, dtype=torch.float32).to('cuda'),
                          ece=torch.tensor(pre_ece, dtype=torch.float32).to('cuda'))
            # checkpoint: model, memory
            utils.save(model, os.path.join(args.save, 'model-weights-valid.pt'))
            utils.pickle_save(memory.state_dict(),
                              os.path.join(args.save, 'memory-warm-up.pickle'))

    # --- Part 2 predictor warm-up ---
    predictor.train()
    for epoch in range(args.predictor_warm_up):
        # warm-up
        if args.num_obj > 1:
            pred_train_loss, (true_acc, true_ece), (pred_acc, pred_ece) = predictor_train(lfs, memory)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                print('[warm-up predictor] epoch %d/%d loss=%.4f' % (epoch, args.predictor_warm_up,
                                                                     pred_train_loss))
                acc_tau = kendalltau(true_acc.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
                ece_tau = kendalltau(true_ece.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
                print('acc kendall\'s-tau=%.4f   ece kendall\'s-tau=%.4f' % (acc_tau, ece_tau))
        else:
            pred_train_loss, true_nll, pred_nll = predictor_train(lfs, memory)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                print('[warm-up predictor] epoch %d/%d loss=%.4f' % (epoch, args.predictor_warm_up, pred_train_loss))
                k_tau = kendalltau(true_nll.detach().to('cpu'), pred_nll.detach().to('cpu'))[0]
                print('kendall\'s-tau=%.4f' % k_tau)
        # save predictor
        utils.save(lfs.predictor, os.path.join(args.save, 'predictor-warm-up.pt'))

    # --- Part 3 loss function search ---
    for epoch in range(args.epochs):
        # search
        pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll = \
            search(train_queue, valid_queue, model, lfs, lossfunc, loss_rejector, optimizer,
                   memory, args.gumbel_scale)
        wandb.log({
            "pre_accuracy": pre_accuracy, "pre_ece": pre_ece, "pre_adaece": pre_adaece, "pre_cece": pre_cece,
            "pre_nll": pre_nll, "T_opt": T_opt, "post_ece": post_ece, "post_adaece": post_adaece,
            "post_cece": post_cece, "post_nll": post_nll,
        }, step=epoch)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))
        # update learning rate
        scheduler.step()


def model_train(train_queue, model, lossfunc, optimizer, name):
    # set model to training model
    model.train()
    # create metrics
    objs = utils.AverageMeter()
    nlls = utils.AverageMeter()
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
        nll, loss = lossfunc(logits, target)
        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        # update metrics
        objs.update(loss.data.item(), n)
        nlls.update(nll.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            print('[%s] train model %03d/%03d loss=%.4f top1-acc=%.4f nll=%.4f' % (
                name, step, total_steps, objs.avg, top1.avg, nlls.avg))
    # return average metrics
    return objs.avg, top1.avg, top5.avg, nlls.avg


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


def predictor_train(lfs, memory):
    objs = utils.AverageMeter()
    batch = memory.get_batch()
    all_acc = []
    all_ece = []
    all_nll = []
    all_pred_acc = []
    all_pred_ece = []
    all_pred_nll = []
    for weights, nll, acc, ece in batch:
        n = nll.size(0)
        if args.num_obj > 1:
            y_pred, predictor_train_loss = lfs.predictor_step(weights, torch.swapaxes(torch.stack([acc, ece]), 0, 1))
            pred_acc, pred_ece = y_pred[:, 0], y_pred[:, 1]
            objs.update(predictor_train_loss.data.item(), n)
            all_acc.append(acc)
            all_ece.append(ece)
            all_nll.append(nll)
            all_pred_acc.append(pred_acc)
            all_pred_ece.append(pred_ece)
        else:
            pred_nll, predictor_train_loss = lfs.predictor_step(weights, nll)
            objs.update(predictor_train_loss.data.item(), n)
            all_nll.append(nll)
            all_pred_nll.append(pred_nll)
    if args.num_obj > 1:
        return objs.avg, (torch.cat(all_acc), torch.cat(all_ece)), (torch.cat(all_pred_acc), torch.cat(all_pred_ece))
    else:
        return objs.avg, torch.cat(all_nll), torch.cat(all_pred_nll)


def search(train_queue, valid_queue, model, lfs, lossfunc, loss_rejector, optimizer, memory, gumbel_scale):
    # -- train model --
    # gumbel sampling and rejection process
    GOOD_LOSS = False
    while not GOOD_LOSS:
        lossfunc.g_ops = gumbel_like(lossfunc.alphas_ops) * gumbel_scale
        GOOD_LOSS, g_ops = loss_rejector.evaluate_loss(lossfunc.g_ops)

    # watch updates
    wandb.config.lossfunc = lossfunc.loss_str(no_gumbel=True)

    print("arch_weights_ops: ", lossfunc.arch_weights_ops())
    print("gumbel_ops: ", lossfunc.g_ops)
    print("alpha_ops: ", F.softmax(lossfunc.alphas_ops, -1))

    # train model for one step
    model_train(train_queue, model, lossfunc, optimizer, name='build memory')
    # -- valid model --
    pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll = model_valid(
        valid_queue, model)
    # save validation to memory
    print('[loss function search] append memory nll=%.4f top1-acc=%.4f ece=%.4f' % (pre_nll, pre_accuracy, pre_ece))
    # save to memory
    memory.append(weights=torch.stack([w.detach() for w in lossfunc.arch_weights()]),
                  nll=torch.tensor(pre_nll, dtype=torch.float32).to('cuda'),
                  acc=torch.tensor(pre_accuracy, dtype=torch.float32).to('cuda'),
                  ece=torch.tensor(pre_ece, dtype=torch.float32).to('cuda'))
    utils.pickle_save(memory.state_dict(),
                      os.path.join(args.save, 'memory-search.pickle'))

    # -- predictor train --
    lfs.predictor.train()
    # use memory to train predictor
    for _ in range(args.predictor_warm_up):
        p_loss, p_true, p_pred = predictor_train(lfs, memory)
        k_tau = kendalltau(p_true.detach().to('cpu'), p_pred.detach().to('cpu'))[0]
        if k_tau > 0.95: break
    print('kendall\'s-tau=%.4f' % k_tau)

    for epoch in range(args.predictor_warm_up):
        if lfs.predictor.num_obj > 1:
            predictor_train_loss, (true_acc, true_ece), (pred_acc, pred_ece) = predictor_train(lfs, memory)
            acc_tau = kendalltau(true_acc.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
            ece_tau = kendalltau(true_ece.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
            if acc_tau > 0.95 and ece_tau > 0.95: break
        else:
            predictor_train_loss, true_nll, pred_nll = predictor_train(lfs, memory)
            k_tau = kendalltau(true_nll.detach().to('cpu'), pred_nll.detach().to('cpu'))[0]
            if k_tau > 0.95: break
    print('kendall\'s-tau=%.4f' % acc_tau if lfs.predictor.num_obj > 1 else k_tau)

    if lfs.predictor.num_obj > 1:
        predictor_pred_loss, (pred_acc, pred_ece) = lfs.step()
    else:
        predictor_pred_loss, pred_nll = lfs.step()
    return pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll


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
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    # search setting
    parser.add_argument('--lfs_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--lfs_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    # load setting
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')

    # loss func setting
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    parser.add_argument('--num_states', type=int, default=11, help='num of operation states')

    # predictor setting
    parser.add_argument('--predictor_type', type=str, default='lstm')
    parser.add_argument('--predictor_warm_up', type=int, default=500, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')

    # others
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # loss function search
    parser.add_argument('--operator_size', type=int, default=8)
    parser.add_argument('--loss_rejector_threshold', type=float, default=0.6, help='loss rejcetion threshold')
    parser.add_argument('--gumbel_scale', type=float, default=1, help='gumbel_scale')
    parser.add_argument('--num_obj', type=int, default=1,
                        help='use multiple objective (acc + lambda * ece) for predictor trianing')
    parser.add_argument('--predictor_lambda', type=float, default=0,
                        help='use multiple objective (acc + lambda * ece) for predictor trianing')
    parser.add_argument('--lfs_lambda', type=float, default=0,
                        help='use multiple objective (acc + lambda * ece) for loss function searching')

    args, unknown_args = parser.parse_known_args()

    args.save = 'checkpoints/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(
        path=args.save,
        scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
    )
    wandb.init(project="Focal Loss Search Calibration", entity="linweitao", config=args)

    main()
