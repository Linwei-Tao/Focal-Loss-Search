import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import functional as F

from module.estimator.gnn import GAEExtractor, GINEncoder, LinearDecoder, ReconstructedLoss
from module.estimator.utils import GraphPreprocessor, arch_matrix_to_graph
from utils import AverageMeter, cal_recon_accuracy, create_exp_dir


def sample_architectures(num_sample, steps=4, num_ops=8, batch_size=64):

    samples = []
    batch_adj_normal = []
    batch_adj_reduce = []
    batch_opt_normal = []
    batch_opt_reduce = []

    k = ((steps + 3) * steps) // 2

    for _ in range(num_sample):
        a_normal = torch.randn(k, num_ops)
        a_reduce = torch.randn(k, num_ops)

        adj_normal, opt_normal = arch_matrix_to_graph(a_normal.unsqueeze(0))
        adj_reduce, opt_reduce = arch_matrix_to_graph(a_reduce.unsqueeze(0))

        batch_adj_normal.append(adj_normal)
        batch_adj_reduce.append(adj_reduce)
        batch_opt_normal.append(opt_normal)
        batch_opt_reduce.append(opt_reduce)

        if len(batch_adj_normal) >= batch_size:
            samples.append((
                (torch.cat(batch_adj_normal), torch.cat(batch_adj_reduce)),
                (torch.cat(batch_opt_normal), torch.cat(batch_opt_reduce))
            ))
            batch_adj_normal = []
            batch_adj_reduce = []
            batch_opt_normal = []
            batch_opt_reduce = []

    if len(batch_adj_normal) > 0:
        samples.append((
            (torch.cat(batch_adj_normal), torch.cat(batch_adj_reduce)),
            (torch.cat(batch_opt_normal), torch.cat(batch_opt_reduce))
        ))

    return samples


def unsup_train(model, data_queue, criterion, optimizer, preprocessor=None, device='cuda', threshold=0.5):
    model.train()
    objs_meter = AverageMeter()
    opt_acc_meter = AverageMeter()
    adj_acc_meter = AverageMeter()

    for step, ((adj_normal, adj_reduce), (opt_normal, opt_reduce)) in enumerate(data_queue):

        n = adj_normal.size(0)

        adj_normal = adj_normal.to(device).requires_grad_(False)
        adj_reduce = adj_reduce.to(device).requires_grad_(False)
        opt_normal = opt_normal.to(device).requires_grad_(False)
        opt_reduce = opt_reduce.to(device).requires_grad_(False)

        if preprocessor is not None:
            processed_adj_normal, processed_opt_normal = preprocessor(adj=adj_normal, opt=opt_normal)
            processed_adj_reduce, processed_opt_reduce = preprocessor(adj=adj_reduce, opt=opt_reduce)
        else:
            processed_adj_normal, processed_opt_normal = adj_normal, opt_normal
            processed_adj_reduce, processed_opt_reduce = adj_reduce, opt_reduce

        optimizer.zero_grad()
        opt_recon_normal, adj_recon_normal, _ = model(opt=processed_opt_normal, adj=processed_adj_normal)
        opt_recon_reduce, adj_recon_reduce, _ = model(opt=processed_opt_reduce, adj=processed_adj_reduce)

        # loss = 0.9 * criterion(inputs=[opt_recon, adj_recon], targets=[opt, adj]) + \
        #        0.1 * ((adj_recon - torch.triu(adj_recon, 1)) ** 2).mean()
        loss = criterion([opt_recon_normal, adj_recon_normal], [opt_normal, adj_normal]) + \
               criterion([opt_recon_reduce, adj_recon_reduce], [opt_reduce, adj_reduce])
        loss.backward()
        optimizer.step()

        opt_acc_normal, adj_acc_normal = cal_recon_accuracy(
            opt=opt_normal, adj=adj_normal, opt_recon=opt_recon_normal, adj_recon=adj_recon_normal, threshold=threshold
        )
        opt_acc_reduce, adj_acc_reduce = cal_recon_accuracy(
            opt=opt_reduce, adj=adj_reduce, opt_recon=opt_recon_reduce, adj_recon=adj_recon_reduce, threshold=threshold
        )

        opt_acc = (opt_acc_normal + opt_acc_reduce) / 2
        adj_acc = (adj_acc_normal + adj_acc_reduce) / 2

        objs_meter.update(loss.data.item(), n)
        opt_acc_meter.update(opt_acc, n)
        adj_acc_meter.update(adj_acc, n)

    return objs_meter.avg, opt_acc_meter.avg, adj_acc_meter.avg


@torch.no_grad()
def unsup_valid(model, data_queue, criterion, preprocessor=None, device='cuda', threshold=0.5):
    model.eval()
    objs_meter = AverageMeter()
    opt_acc_meter = AverageMeter()
    adj_acc_meter = AverageMeter()

    for step, ((adj_normal, adj_reduce), (opt_normal, opt_reduce)) in enumerate(data_queue):

        n = adj_normal.size(0)

        adj_normal = adj_normal.to(device).requires_grad_(False)
        adj_reduce = adj_reduce.to(device).requires_grad_(False)
        opt_normal = opt_normal.to(device).requires_grad_(False)
        opt_reduce = opt_reduce.to(device).requires_grad_(False)

        if preprocessor is not None:
            processed_adj_normal, processed_opt_normal = preprocessor(adj=adj_normal, opt=opt_normal)
            processed_adj_reduce, processed_opt_reduce = preprocessor(adj=adj_reduce, opt=opt_reduce)
        else:
            processed_adj_normal, processed_opt_normal = adj_normal, opt_normal
            processed_adj_reduce, processed_opt_reduce = adj_reduce, opt_reduce

        opt_recon_normal, adj_recon_normal, _ = model(opt=processed_opt_normal, adj=processed_adj_normal)
        opt_recon_reduce, adj_recon_reduce, _ = model(opt=processed_opt_reduce, adj=processed_adj_reduce)

        loss = criterion([opt_recon_normal, adj_recon_normal], [opt_normal, adj_normal]) + \
               criterion([opt_recon_reduce, adj_recon_reduce], [opt_reduce, adj_reduce])

        opt_acc_normal, adj_acc_normal = cal_recon_accuracy(
            opt=opt_normal, adj=adj_normal, opt_recon=opt_recon_normal, adj_recon=adj_recon_normal, threshold=threshold
        )
        opt_acc_reduce, adj_acc_reduce = cal_recon_accuracy(
            opt=opt_reduce, adj=adj_reduce, opt_recon=opt_recon_reduce, adj_recon=adj_recon_reduce, threshold=threshold
        )

        opt_acc = (opt_acc_normal + opt_acc_reduce) / 2
        adj_acc = (adj_acc_normal + adj_acc_reduce) / 2

        objs_meter.update(loss.data.item(), n)
        opt_acc_meter.update(opt_acc, n)
        adj_acc_meter.update(adj_acc, n)

    return objs_meter.avg, opt_acc_meter.avg, adj_acc_meter.avg


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # enable GPU and set random seeds
    np.random.seed(args.seed)                  # set random seed: numpy

    # NOTE: "deterministic" and "benchmark" are set for reproducibility
    # such settings have impacts on efficiency
    # for speed test, disable "deterministic" and enable "benchmark"
    # reproducible search
    # cudnn.deterministic = True
    # cudnn.benchmark = False
    # fast search
    cudnn.deterministic = False
    cudnn.benchmark = True

    torch.manual_seed(args.seed)               # set random seed: torch
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)          # set random seed: torch.cuda

    logging.info('args: %s' % args)
    if len(unknown_args) > 0:
        logging.warning('unknown_args: %s' % unknown_args)
    else:
        logging.info('unknown_args: %s' % unknown_args)

    # -- check gpu --
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # -- load data --
    # TODO: sample architectures
    unsup_queue = sample_architectures(num_sample=4000)
    valid_queue = sample_architectures(num_sample=1000)

    # -- preprocessor --
    preprocessor = GraphPreprocessor(mode=args.preprocess_mode, lamb=args.preprocess_lamb)

    # -- build model --
    model = GAEExtractor(
        encoder=GINEncoder(
            input_dim=args.opt_num, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
            num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers
        ),
        decoder=LinearDecoder(
            latent_dim=args.latent_dim, decode_dim=args.opt_num, dropout=args.dropout,
            activation_adj=torch.sigmoid, activation_opt=torch.softmax
        )
    )
    model = model.to(device)

    # -- loss function & optimizer--
    # criterion = ReconstructedLoss(loss_opt=F.mse_loss, loss_adj=F.mse_loss, w_opt=1.0, w_adj=1.0)
    criterion = ReconstructedLoss(loss_opt=F.mse_loss, loss_adj=torch.nn.BCELoss(), w_opt=1.0, w_adj=1.0)
    gae_optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr,
                                     betas=(args.beta_0, args.beta_1), eps=args.eps)

    best_epoch = 0
    best_objs_valid = float('inf')

    # -- training loop --
    for ep in range(1, args.epochs + 1):

        objs_train, opt_acc_train, adj_acc_train = unsup_train(
            model=model, data_queue=unsup_queue, criterion=criterion, optimizer=gae_optimizer,
            preprocessor=preprocessor, device=device, threshold=args.threshold
        )
        objs_valid, opt_acc_valid, adj_acc_valid = unsup_valid(
            model=model, data_queue=valid_queue, criterion=criterion,
            preprocessor=preprocessor, device=device, threshold=args.threshold
        )

        logging.info(
            'Ep %03d/%03d [unsupervised]: train loss=%.4f, opt_acc=%5.2f, adj_acc=%5.2f; '
            'valid loss=%.4f, opt_acc=%5.2f, adj_acc=%5.2f'
            % (ep, args.epochs, objs_train, opt_acc_train, adj_acc_train, objs_valid, opt_acc_valid, adj_acc_valid)
        )

        if objs_valid < best_objs_valid:
            torch.save(
                {
                    'weights': model.state_dict(),
                    'epoch': ep,
                    'preprocessor': {
                        'mode': args.preprocess_mode,
                        'lamb': args.preprocess_lamb
                    },
                    'objs_valid': objs_valid,
                    'opt_acc_valid': opt_acc_valid,
                    'adj_acc_valid': adj_acc_valid,
                },
                os.path.join(save_path, 'best_tau_ckpt.pth.tar')
            )
            best_epoch = ep
            best_objs_valid = objs_valid

    logging.info('best model with objs_valid=%.4f at ep=%d' % (best_objs_valid, best_epoch))


if __name__ == '__main__':

    # -- setup --
    parser = argparse.ArgumentParser('gae_nb_train_sep')
    # network
    parser.add_argument('--opt_num', type=int, default=11)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.5)
    # optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta_0', type=float, default=0.5)
    parser.add_argument('--beta_1', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=3e-4)
    # data
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--preprocess_mode', type=int, default=4)
    parser.add_argument('--preprocess_lamb', type=float, default=0.)
    # others
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False, help='debug logging')

    args, unknown_args = parser.parse_known_args()

    # -- set save path --
    save_path = 'checkpoints/train-extractor-%s' % (time.strftime("%Y%m%d-%H%M%S"),)
    create_exp_dir(path=save_path, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(levelname)s %(message)s'
    logging_level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(stream=sys.stdout, level=logging_level,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    main()
