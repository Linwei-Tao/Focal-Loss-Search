import torch
import torch.utils
import utils


def predictor_train(lfs, memory, args):
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
        weights = weights.cuda(non_blocking=True)
        nll = nll.cuda(non_blocking=True)
        acc = acc.cuda(non_blocking=True)
        ece = ece.cuda(non_blocking=True)

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
