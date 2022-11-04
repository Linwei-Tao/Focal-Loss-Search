from typing import Union, List

import torch
from torch import nn
from module.operations import OPS
from genotypes import PRIMITIVES, SEARCHED_LOSS
from utils import gumbel_like, gumbel_softmax
import torch.nn.functional as F


class MixedOp(nn.Module):

    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive]()
            self._ops.append(op)

    def forward(self, x1, x2, ops_weights, gumbel_training=True):
        idx = ops_weights.argmax(dim=-1)
        if gumbel_training:
            return ops_weights[idx] * self._ops[idx](x1, x2)
        else:
            return self._ops[idx](x1, x2)


class LossFunc(nn.Module):

    def __init__(self, num_states=11, tau=0.1, noCEFormat=True, searched_loss=None):
        super(LossFunc, self).__init__()
        self.num_initial_state = 3  # 1, p_k, p_j
        self.states = []
        self.num_states = num_states
        self._tau = tau
        self.noCEFormat = noCEFormat
        self.searched_loss = searched_loss

        self._ops = nn.ModuleList()
        for i in range(num_states):
            op = MixedOp()
            self._ops.append(op)

        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        if searched_loss:
            self.alphas_ops = SEARCHED_LOSS[searched_loss].to('cuda')
        else:
            self.alphas_ops = (1e-3 * torch.randn(num_states, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_ops = gumbel_like(self.alphas_ops)

    def forward(self, logits, target, output_loss_array=False, gumbel_training=True):
        target = target.view(-1, 1)
        logp_k = F.log_softmax(logits, -1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()  # p_k: probility at target label
        p_j_mask = torch.lt(softmax_logits,
                            p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()
        if gumbel_training:
            ops_weights = gumbel_softmax(self.alphas_ops.data, tau=self._tau, dim=-1, g=self.g_ops)
        else:
            ops_weights = F.softmax(self.alphas_ops, -1)
        self.states = [p_k, p_j]
        self.states_op_record = [p_k, p_j]

        s0, s1 = p_k, p_j
        for i in range(self.num_states):
            s0, s1 = s1, self._ops[i](s0, s1, ops_weights[i], gumbel_training=gumbel_training)
            self.states.append(s1)
        nll = -logp_k
        # output xxxx * -log(p_k) or not
        if self.noCEFormat:
            loss = self.states[-1]
        else:
            loss = self.states[-1] * nll

        if output_loss_array:
            return loss, nll
        return loss.sum(), nll.sum()

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_ops]

    def arch_weights(self, cat: bool = False) -> Union[List[torch.tensor], torch.tensor]:
        weights = gumbel_softmax(self.alphas_ops, tau=self._tau, dim=-1, g=self.g_ops)
        if cat:
            return torch.cat(weights, dim=-1)
        else:
            return weights

    def arch_weights_ops(self) -> Union[List[torch.tensor], torch.tensor]:
        return gumbel_softmax(self.alphas_ops, tau=self._tau, dim=-1, g=self.g_ops)

    def arch_weights_operators(self) -> Union[List[torch.tensor], torch.tensor]:
        return gumbel_softmax(self.alphas_operators, tau=self._tau, dim=-1, g=self.g_operators)

    def loss_str(self, return_records=False, no_gumbel=False):
        if no_gumbel:
            ops_weights = self.alphas_ops.data
        else:
            ops_weights = gumbel_softmax(self.alphas_ops.data, tau=self._tau, dim=-1, g=self.g_ops)

        states = ["p_k", "p_j"]
        op_list = []
        for i in range(self.num_states):
            idx = ops_weights[i].argmax(dim=-1)
            op_list.append(PRIMITIVES[idx])

        s0, s1 = "p_k", "p_j"
        for index, op in enumerate(op_list):
            s0, s1 = self.op_str(op, s0, s1)
            states.append(s1)
        if self.noCEFormat:
            out_str = states[-1]
        else:
            out_str = "-({})*log(p_k)".format(states[-1])
        if return_records == True:
            return out_str, states
        return out_str

    def op_str(self, op, s0, s1):
        if op == 'add':
            s0, s1 = s1, "{} + {}".format(s0, s1)
        elif op == 'mul':
            s0, s1 = s1, "mul({}, {})".format(s0, s1)
        elif op == 'iden1':
            s0, s1 = s1, s0
        elif op == 'one_plus':
            s0, s1 = s1, "1 + {}".format(s0)
        elif op == 'one_minus':
            s0, s1 = s1, "1 - {}".format(s0)
        else:
            s0, s1 = s1, "{}({})".format(op, s0)
        return s0, s1
