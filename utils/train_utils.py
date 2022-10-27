import torch.nn as nn
import utils

def model_train(train_queue, model, lossfunc, optimizer, name, args, gumbel_training=True):
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
        nll, loss = lossfunc(logits, target, gumbel_training=gumbel_training)
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