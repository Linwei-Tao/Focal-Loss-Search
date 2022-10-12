from typing import Union, List

import torch
from torch import nn, autograd
from module.operations import FactorizedReduce, OPS, ReLUConvBN
from genotypes import Genotype, PRIMITIVES
from utils import gumbel_like
from utils import gumbel_softmax_v1 as gumbel_softmax
from collections import deque
import torch.nn.functional as F



class MixedOp(nn.Module):

    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive]()
            self._ops.append(op)

    def forward(self, states, operator_weights, ops_weights):
        idx = ops_weights.argmax(dim=-1)
        x1_idx, x2_idx = operator_weights.topk(k=2, dim=-1)[1]
        return ops_weights[idx] * self._ops[idx](states[x1_idx], states[x2_idx]), self._ops[idx], x1_idx, x2_idx

        # return sum(w * op(x) for w, op in zip(weights, self._ops))


class LossFunc(nn.Module):

    def __init__(self, num_operator_choice=8, num_states=11, tau=0.1, gamma=5):
        super(LossFunc, self).__init__()
        self.num_initial_state = 3 # 1, p_k, p_j
        self.states = []
        self.num_operator_choice = num_operator_choice
        self.num_states = num_states
        self._tau = tau
        self.gamma = gamma

        self._ops = nn.ModuleList()
        for i in range(num_states):
            op = MixedOp()
            self._ops.append(op)

        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        self.alphas_ops = (1e-3 * torch.randn(num_states, num_ops)).to('cuda').requires_grad_(True)
        self.alphas_operators = (1e-3 * torch.randn(num_states, num_operator_choice)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_ops = gumbel_like(self.alphas_ops)
        self.g_operators = gumbel_like(self.alphas_operators)

    def forward(self, logits, target):
        target = target.view(-1, 1)
        logp_k = F.log_softmax(logits, -1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits,
                            p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()


        ops_weights = gumbel_softmax(self.alphas_ops.data, tau=self._tau, dim=-1, g=self.g_ops)
        operator_weights = gumbel_softmax(self.alphas_operators.data, tau=self._tau, dim=-1, g=self.g_operators)

        self.states = [torch.ones_like(p_k, requires_grad=True), p_k, p_j]
        while len(self.states) < self.num_operator_choice:
            self.states.extend(self.states)

        operator_choices = self.states[-self.num_operator_choice:]
        op_list = []
        x1_list = []
        for i in range(self.num_states):
            s, op, x1_idx, x2_idx = self._ops[i](operator_choices, operator_weights[i], ops_weights[i])
            self.states.append(s)
        loss = -self.states[-1].pow(self.gamma) * logp_k
        return loss.sum()

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_ops, self.alphas_operators]

class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, tau, steps=4, multiplier=4, stem_multiplier=3):
        """
        :param C: init channels number
        :param num_classes: classes numbers
        :param layers: total number of layers
        :param criterion: loss function
        :param steps:
        :param multiplier:
        :param stem_multiplier:
        """
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._tau = tau

        # stem layer
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # body layers (normal and reduction)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to('cuda')
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = gumbel_softmax(self.alphas_reduce.data, tau=self._tau, dim=-1, g=self.g_reduce)
            else:
                weights = gumbel_softmax(self.alphas_normal.data, tau=self._tau, dim=-1, g=self.g_normal)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # calculate edge number k = (((1+1) + (steps+1)) * steps) / 2
        k = ((self._steps + 3) * self._steps) // 2
        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_normal = gumbel_like(self.alphas_normal)
        self.g_reduce = gumbel_like(self.alphas_reduce)

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_normal, self.alphas_reduce]

    def arch_weights(self, cat: bool = True) -> Union[List[torch.tensor], torch.tensor]:
        weights = [
            gumbel_softmax(self.alphas_normal, tau=self._tau, dim=-1, g=self.g_normal),
            gumbel_softmax(self.alphas_reduce, tau=self._tau, dim=-1, g=self.g_reduce)
        ]
        if cat:
            return torch.cat(weights)
        else:
            return weights

    def genotype(self) -> Genotype:

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(self.alphas_normal.detach().to('cpu').numpy())
        gene_reduce = _parse(self.alphas_reduce.detach().to('cpu').numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
