from typing import Union, List

import torch
from torch import nn
from module.operations import OPS
from genotypes import PRIMITIVES
from utils import gumbel_like
from utils import gumbel_softmax
import torch.nn.functional as F


class MixedOp(nn.Module):

    def __init__(self):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive]()
            self._ops.append(op)

    def forward(self, x1, x2, ops_weights):
        idx = ops_weights.argmax(dim=-1)
        return ops_weights[idx] * self._ops[idx](x1, x2)

        # return sum(w * op(x) for w, op in zip(weights, self._ops))


class LossFunc(nn.Module):

    def __init__(self, num_states=11, tau=0.1):
        super(LossFunc, self).__init__()
        self.num_initial_state = 3  # 1, p_k, p_j
        self.states = []
        self.num_states = num_states
        self._tau = tau

        self._ops = nn.ModuleList()
        for i in range(num_states):
            op = MixedOp()
            self._ops.append(op)

        # operations number
        num_ops = len(PRIMITIVES)

        # init architecture parameters alpha
        self.alphas_ops = (1e-3 * torch.randn(num_states, num_ops)).to('cuda').requires_grad_(True)
        # init Gumbel distribution for Gumbel softmax sampler
        self.g_ops = gumbel_like(self.alphas_ops)

    def forward(self, logits, target, output_loss_array=False):
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
        self.states = [p_k, p_j]
        self.states_op_record = [p_k, p_j]

        s0, s1 = p_k, p_j
        for i in range(self.num_states):
            s0, s1 = s1, self._ops[i](s0, s1, ops_weights[i])
            self.states.append(s1)
        nll = -logp_k
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
        if return_records == True:
            return "-({})*log(p_k)".format(states[-1]), states
        return "-({})*log(p_k)".format(states[-1])

    def op_str(self, op, s0, s1):
        if op == 'add':
            s0, s1 = s1, "{} + {}".format(s0, s1)
        elif op == 'mul':
            s0, s1 = s1, "{} * {}".format(s0, s1)
        elif op == 'iden1':
            s0, s1 = s1, s0
        elif op == 'one_plus':
            s0, s1 = s1, "1 + {}".format(s0)
        elif op == 'one_minus':
            s0, s1 = s1, "1 - {}".format(s0)
        else:
            s0, s1 = s1, "{}({})".format(op, s0)
        return s0, s1