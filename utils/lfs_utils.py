import os
import torch
import torch.utils
from scipy.stats import kendalltau
from torch.nn import functional as F

from utils.train_utils import model_train
from utils.valid_utils import model_valid
from utils.predictor_utils import predictor_train

# Import utilities
from utils import gumbel_like, pickle_save
import utils


def search(train_queue, valid_queue, model, lfs, lossfunc, loss_rejector, optimizer, memory, gumbel_scale, args, epoch):
    # -- train model --
    # gumbel sampling and rejection process
    GOOD_LOSS = False
    while not GOOD_LOSS:
        lossfunc.g_ops = gumbel_like(lossfunc.alphas_ops) * gumbel_scale
        GOOD_LOSS, g_ops = loss_rejector.evaluate_loss(lossfunc.g_ops)

    # train model for one step
    model_train(train_queue, model, lossfunc, optimizer, name='[{}] Searching {}/{}'.format(args.device, epoch+1, args.search_epochs), args=args)
    # -- valid model --
    pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll = model_valid(
        valid_queue, valid_queue, model)
    # save validation to memory
    print('[Loss Function Search] append memory NLL=%.4f ACC=%.4f ECE=%.4f' % (pre_nll, pre_accuracy, pre_ece))
    # save to memory
    memory.append(weights=torch.stack([w.detach() for w in lossfunc.arch_weights()]),
                  nll=torch.tensor(pre_nll, dtype=torch.float32).to('cuda'),
                  acc=torch.tensor(pre_accuracy, dtype=torch.float32).to('cuda'),
                  ece=torch.tensor(pre_ece, dtype=torch.float32).to('cuda'))

    # -- predictor train --
    lfs.predictor.train()

    # use memory to train predictor
    k_tau = 0
    for epoch in range(args.predictor_warm_up):
        if args.num_obj > 1:
            predictor_train_loss, (true_acc, true_ece), (pred_acc, pred_ece) = predictor_train(lfs, memory, args)
            acc_tau = kendalltau(true_acc.detach().to('cpu'), pred_acc.detach().to('cpu'))[0]
            ece_tau = kendalltau(true_ece.detach().to('cpu'), pred_ece.detach().to('cpu'))[0]
            if acc_tau > 0.95 and ece_tau > 0.95: break
            k_tau = acc_tau
        else:
            predictor_train_loss, true_nll, pred_nll = predictor_train(lfs, memory, args)
            k_tau = kendalltau(true_nll.detach().to('cpu'), pred_nll.detach().to('cpu'))[0]
            if k_tau > 0.95: break
    print('kendall\'s-tau=%.4f' % k_tau)

    lfs.step()

    return pre_accuracy, pre_ece, pre_adaece, pre_cece, pre_nll, T_opt, post_ece, post_adaece, post_cece, post_nll, lossfunc.loss_str(), lossfunc.loss_str(
        no_gumbel=True)
