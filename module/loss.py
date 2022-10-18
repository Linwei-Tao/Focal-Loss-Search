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

    def forward(self, states, operator_weights, ops_weights):
        idx = ops_weights.argmax(dim=-1)
        x1_idx, x2_idx = operator_weights.topk(k=2, dim=-1)[1]
        return ops_weights[idx] * self._ops[idx](states[x1_idx], states[x2_idx]), self._ops[idx], x1_idx, x2_idx

        # return sum(w * op(x) for w, op in zip(weights, self._ops))


class LossFunc(nn.Module):

    def __init__(self, num_operator_choice=8, num_states=11, tau=0.1, gamma=5):
        super(LossFunc, self).__init__()
        self.num_initial_state = 3  # 1, p_k, p_j
        self.states = []
        self.num_operator_choice = num_operator_choice
        self.num_states = num_states
        self._tau = tau
        self.gamma = gamma  # fix gamma

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
        self.states = [torch.ones_like(p_k, requires_grad=True), p_k, p_j, torch.ones_like(p_k, requires_grad=True), p_k, p_j, p_k, p_j]

        # while len(self.states) < self.num_operator_choice:
        #     self.states.extend(self.states)

        operator_choices = self.states[-self.num_operator_choice:]
        for i in range(self.num_states):
            s, op, x1_idx, x2_idx = self._ops[i](operator_choices, operator_weights[i], ops_weights[i])
            self.states.append(s)
        # loss = -self.states[-1].pow(self.gamma) * logp_k
        nll = -logp_k
        loss = self.states[-1] * nll
        return loss.sum(), nll.sum()

    def arch_parameters(self) -> List[torch.tensor]:
        return [self.alphas_ops, self.alphas_operators]

    def arch_weights(self, cat: bool = True) -> Union[List[torch.tensor], torch.tensor]:
        weights = [
            gumbel_softmax(self.alphas_ops, tau=self._tau, dim=-1, g=self.g_ops),
            gumbel_softmax(self.alphas_operators, tau=self._tau, dim=-1, g=self.g_operators)
        ]
        if cat:
            return torch.cat(weights, dim=-1)
        else:
            return weights

    def arch_weights_ops(self, cat: bool = True) -> Union[List[torch.tensor], torch.tensor]:
        return gumbel_softmax(self.alphas_ops, tau=self._tau, dim=-1, g=self.g_ops)

    def arch_weights_operators(self, cat: bool = True) -> Union[List[torch.tensor], torch.tensor]:
        return gumbel_softmax(self.alphas_operators, tau=self._tau, dim=-1, g=self.g_operators)



    def loss_str(self):
        ops_weights = gumbel_softmax(self.alphas_ops.data, tau=self._tau, dim=-1, g=self.g_ops)
        operator_weights = gumbel_softmax(self.alphas_operators.data, tau=self._tau, dim=-1, g=self.g_operators)
        states = ["1", "p_k", "p_j", "1", "p_k", "p_j", "p_k", "p_j"]

        op_list = []
        operator_list = []
        for i in range(self.num_states):
            idx = ops_weights[i].argmax(dim=-1)
            op_list.append(PRIMITIVES[idx])
            x1_idx, x2_idx = operator_weights[i].topk(k=2, dim=-1)[1]
            operator_list.append((x1_idx, x2_idx))

        for index, op in enumerate(op_list):
            x1_idx, x2_idx = operator_list[index]
            if op == 'add' or op == 'mul':
                s = "{}({}, {})".format(op, states[-8:][x1_idx], states[-8:][x2_idx])
            else:
                s = "{}({})".format(op, states[-8:][x1_idx])
            states.append(s)
        return "-{}*log(p_k)".format(states[-1])
