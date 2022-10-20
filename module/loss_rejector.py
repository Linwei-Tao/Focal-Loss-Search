import torch
from torch import nn
import utils
from torch.nn import functional as F


class LossRejector(nn.Module):

    def __init__(self, lossfunc, train_queue, model, num_rejection_sample=5, threshold=0.9,
                 Gradient_Equivalence_Check=True):
        super(LossRejector, self).__init__()
        self.lossfunc = lossfunc
        self.train_queue = train_queue
        self.model = model
        self.lr = 0.001
        self.momentum = 0.9
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        self.steps = 500

        self.num_rejection_sample = num_rejection_sample  # current code do not use this argument, we evaluate the loss with a batch of samples
        self.threshold = threshold
        self.Gradient_Equivalence_Check = Gradient_Equivalence_Check
        self.Gradient_Equivalence = []

    def evaluate_loss(self, g_ops):
        x, target = next(iter(self.train_queue))
        n = x.size(0)
        # data to CUDA
        x = x.to('cuda').requires_grad_(False)
        target = target.to('cuda', non_blocking=True).requires_grad_(False)
        random_logits = self.model(x)
        random_prec1, random_prec5 = utils.accuracy(random_logits, target, topk=(1, 5))

        # update logits
        learnable_logits = nn.Parameter(random_logits.data)
        self.optimizer.param_groups[0]['params'] = learnable_logits
        self.optimizer = torch.optim.SGD([learnable_logits], lr=self.lr, momentum=self.momentum)

        # assign gumbels to lossfunc
        self.lossfunc.g_ops = g_ops

        # check1: loss at pk=0 equals to 0 ===========> with log(p_k), the loss function is innative 0

        # check2: Monotonically decreasing between 0～1
        # test 5 logits, need the rank align with input rank. Softmax logits 0.9 0.7 0.5 0.3 0.1 monotonically decreasing
        test_logit = torch.FloatTensor(
            [[5.5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [4.1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [3.2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [2.3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1.1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        ).cuda()


        # using first dimensional as p_k
        test_target = torch.zeros(test_logit.shape[0], dtype=torch.int64).cuda()
        _, logits_rank = F.softmax(test_logit, -1)[:, 0].sort(descending=True)
        loss_array, nll_array = self.lossfunc(test_logit, test_target, output_loss_array=True)
        if not (loss_array.sort()[1] == logits_rank).sum() == logits_rank.shape[0]:
            return 0, None

        # check3: optimization performance higher than a threhold
        for step in range(self.steps):
            self.optimizer.zero_grad()
            loss, nll = self.lossfunc(learnable_logits, target)
            # backward
            loss.backward()
            self.optimizer.step()
        learnd_prec1, learnd_prec5 = utils.accuracy(learnable_logits, target, topk=(1, 5))
        if learnd_prec1 / 100 > self.threshold:
            print("Got One Loss with acc {:2f}%!!!  {}".format(learnd_prec1, self.lossfunc.loss_str()))
            return 1, g_ops
        else:
            return 0, None
