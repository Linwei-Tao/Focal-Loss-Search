import argparse
import glob
import wandb
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
import torch.utils
from scipy.stats import kendalltau
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

from module.loss_searcher import LFS
from module.memory import Memory
from module.predictor import Predictor

from module.resnet import resnet50
from module.loss import LossFunc
from module.loss_rejector import LossRejector

# Import utilities
from utils import gumbel_like, MO_MSE

from utils.predictor_utils import predictor_train
from utils.train_utils import model_train
from utils.valid_utils import model_valid
from utils.lfs_utils import search
from utils.retrain_utils import retrain

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
    wandb.config.searched_loss_str = 0

    # use SGD to optimize the model
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # construct data transformer (including normalization, augmentation)
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    # load CIFAR10 data training set (train=True)
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
    scheduler = CosineAnnealingLR(optimizer, int(args.search_epochs), eta_min=args.learning_rate_min)

    # loss function
    lossfunc = LossFunc(num_states=args.num_states, tau=args.tau, noCEFormat=args.noCEFormat)

    # a loss evaluator that filter unpromising loss function
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
            if flag: warm_up_gumbel.append(g_ops)
        utils.pickle_save(warm_up_gumbel, os.path.join(args.save, 'gumbel-warm-up.pickle'))
        # 1.1.2 warm up
        for epoch, gumbel in enumerate(warm_up_gumbel):
            # warm-up
            lossfunc.g_ops = gumbel
            print("Objective function: %s" % (lossfunc.loss_str()))
            objs, top1, top5, nll = model_train(train_queue, model, lossfunc, optimizer,
                                                name='Warm Up Epoch {}/{}'.format(epoch + 1, args.warm_up_population),
                                                args=args)
            print('[Warm Up Epoch {}/{}] searched loss={} top1-acc={} nll={}'.format(
                epoch + 1, args.warm_up_population, objs, top1, nll))
            # save weights
            utils.save(model, os.path.join(args.save, 'model-weights-warm-up.pt'))

    # 1.2 build memory (i.e. valid model)
    if args.load_memory is not None:
        print('Load valid model from {}'.format(args.load_model))
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
            model_train(train_queue, model, lossfunc, optimizer,
                        name='Build Memory Epoch {}/{}'.format(epoch + 1, args.warm_up_population), args=args)
            # valid model
            pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, \
            post_cece, post_nll = model_valid(valid_queue, valid_queue, model)
            print('[Build Memory Epoch {}/{}] valid model-{} valid_nll={} valid_acc={} valid_ece={}'.format(epoch + 1,
                                                                                                            args.warm_up_population,
                                                                                                            epoch + 1,
                                                                                                            pre_nll,
                                                                                                            pre_accuracy,
                                                                                                            pre_ece))
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
    # -- build predictor --
    _, feature_num = torch.cat(lossfunc.arch_parameters()).shape

    predictor = Predictor(input_size=feature_num,
                          hidden_size=args.predictor_hidden_state,
                          num_obj=args.num_obj,
                          predictor_lambda=args.predictor_lambda).cuda()

    # -- build loss function searcher --
    lfs_criterion = MO_MSE(args.lfs_lambda) if args.num_obj > 1 else F.mse_loss
    predictor_criterion = MO_MSE(args.predictor_lambda) if args.num_obj > 1 else F.mse_loss
    lfs = LFS(
        lossfunc=lossfunc, model=model, momentum=args.momentum, weight_decay=args.weight_decay,
        lfs_learning_rate=args.lfs_learning_rate, lfs_weight_decay=args.lfs_weight_decay,
        predictor=predictor, pred_learning_rate=args.pred_learning_rate,
        lfs_criterion=lfs_criterion, predictor_criterion=predictor_criterion
    )

    # -- train predictor--
    predictor.train()
    for epoch in range(args.predictor_warm_up):
        # warm-up
        if args.num_obj > 1:
            pred_train_loss, (true_acc, true_ece), (pred_acc, pred_ece) = predictor_train(lfs, memory, args)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                print('[Warm up Predictor Epoch {}/{}]  loss={}'.format(epoch, args.predictor_warm_up,
                                                                        pred_train_loss))
                acc_tau = kendalltau(true_acc.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
                ece_tau = kendalltau(true_ece.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
                print('acc kendall\'s-tau={} ece kendall\'s-tau={}'.format(acc_tau, ece_tau))
        else:
            pred_train_loss, true_nll, pred_nll = predictor_train(lfs, memory, args)
            if epoch % args.report_freq == 0 or epoch == args.predictor_warm_up:
                print('[Warm up Predictor Epoch {}/{}] loss={}'.format(epoch, args.predictor_warm_up, pred_train_loss))
                k_tau = kendalltau(true_nll.detach().to('cpu'), pred_nll.detach().to('cpu'))[0]
                print('kendall\'s-tau={}'.format(k_tau))
        # save predictor
        utils.save(lfs.predictor, os.path.join(args.save, 'predictor-warm-up.pt'))

    # --- Part 3 loss function search ---
    for epoch in range(args.search_epochs):
        print(lossfunc.loss_str(no_gumbel=True))
        print(lossfunc.alphas_ops)
        print(F.softmax(lossfunc.alphas_ops,-1))

        # search
        pre_valid_accuracy, pre_valid_ece, pre_valid_adaece, pre_valid_cece, pre_valid_nll, T_opt, post_valid_ece, \
        post_valid_adaece, post_valid_cece, post_valid_nll, gumbel_loss_str, searched_loss_str = \
            search(train_queue, valid_queue, model, lfs, lossfunc, loss_rejector, optimizer,
                   memory, args.gumbel_scale, args, epoch)
        wandb.config.update({"searched_loss_str": searched_loss_str}, allow_val_change=True)
        wandb.log({
            "search_pre_valid_accuracy": pre_valid_accuracy * 100, "search_pre_valid_ece": pre_valid_ece * 100,
            "search_pre_valid_adaece": pre_valid_adaece * 100, "search_pre_valid_cece": pre_valid_cece * 100,
            "search_pre_valid_nll": pre_valid_nll * 100, "search_T_opt": T_opt,
            "search_post_valid_ece": post_valid_ece * 100,
            "search_post_valid_adaece": post_valid_adaece * 100,
            "search_post_valid_cece": post_valid_cece * 100, "search_post_valid_nll": post_valid_nll * 100,
        }, step=epoch)
        # save weights
        utils.save(model, os.path.join(args.save, 'model-weights-search.pt'))
        # update learning rate
        scheduler.step()

    # --- Part 4 retrain on searched loss ---
    if args.retrain_epochs > 0:
        retrain(model, lossfunc, args, wandb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Loss Function Search")
    # data
    parser.add_argument('--data', type=str, default='/data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10')
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
    parser.add_argument('--search_epochs', type=int, default=200, help='num of searching epochs')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    # search setting
    parser.add_argument('--lfs_learning_rate', type=float, default=1e-1, help='learning rate for arch encoding')
    parser.add_argument('--lfs_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    # load setting
    parser.add_argument('--load_model', type=str, default=None, help='load model weights from file')
    parser.add_argument('--load_memory', type=str, default=None, help='load memory from file')
    parser.add_argument('--load_checkpoints', action='store_true', default=False, help='use both model and memory')

    # loss func setting
    parser.add_argument('--tau', type=float, default=0.1, help='tau')
    parser.add_argument('--num_states', type=int, default=11, help='num of operation states')
    parser.add_argument('--noCEFormat', action='store_true', default=False, help='not use SEARCHLOSS * -log(p_k)')

    # predictor setting
    parser.add_argument('--predictor_warm_up', type=int, default=2000, help='predictor warm-up steps')
    parser.add_argument('--predictor_hidden_state', type=int, default=16, help='predictor hidden state')
    parser.add_argument('--predictor_batch_size', type=int, default=64, help='predictor batch size')
    parser.add_argument('--pred_learning_rate', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--pred_weight_decay', type=float, default=1e-3, help='predictor learning rate')
    parser.add_argument('--memory_size', type=int, default=100, help='size of memory to train predictor')
    parser.add_argument('--warm_up_population', type=int, default=100, help='warm_up_population')

    # others
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=int, default=0)


# loss function search
    parser.add_argument('--operator_size', type=int, default=8)
    parser.add_argument('--loss_rejector_threshold', type=float, default=0.6, help='loss rejection threshold')
    parser.add_argument('--gumbel_scale', type=float, default=1, help='gumbel_scale')
    parser.add_argument('--num_obj', type=int, default=1,
                        help='use multiple objective (acc + lambda * ece) for predictor training')
    parser.add_argument('--predictor_lambda', type=float, default=None,
                        help='use multiple objective (acc + lambda * ece) for predictor training')
    parser.add_argument('--lfs_lambda', type=float, default=None,
                        help='use multiple objective (acc + lambda * ece) for loss function searching')

    # retrain
    parser.add_argument('--retrain_epochs', type=int, default=350, help='retrain epochs')

    args, unknown_args = parser.parse_known_args()

    args.save = 'checkpoints/search_retrain-n_state{}-{}'.format(args.num_states, np.random.randint(100000))
    utils.create_exp_dir(
        path=args.save,
        scripts_to_save=glob.glob('*.py') + glob.glob('module/**/*.py', recursive=True)
    )

    # load dir
    if args.load_checkpoints:
        args.load_model = 'checkpoints/n_states={}'.format(args.num_states)
        args.load_memory = 'checkpoints/n_states={}'.format(args.num_states)


    wandb.init(project="Focal Loss Search Calibration", entity="linweitao", config=args)

    main()
