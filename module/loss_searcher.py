import torch

from torch.nn import functional as F

class LFS(object):

    def __init__(self, lossfunc, model, momentum, weight_decay,
                 lfs_learning_rate, lfs_weight_decay,
                 predictor, pred_learning_rate,
                 lfs_criterion=F.mse_loss,
                 predictor_criterion=F.mse_loss):
        self.network_momentum = momentum
        self.network_weight_decay = weight_decay

        # models
        self.lossfunc = lossfunc
        self.model = model
        self.predictor = predictor

        # lfs optimization
        self.lfs_optimizer = torch.optim.Adam(
            self.lossfunc.arch_parameters(), lr=lfs_learning_rate, betas=(0.5, 0.999),
            weight_decay=lfs_weight_decay
        )
        self.lfs_criterion = lfs_criterion

        # predictor optimization
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=pred_learning_rate, betas=(0.5, 0.999)
        )
        self.predictor_criterion = predictor_criterion


    def predictor_step(self, x, y):
        # clear prev gradient
        self.predictor_optimizer.zero_grad()
        # get output
        y_pred = self.predictor(x)
        # calculate loss
        loss = self.predictor_criterion(y_pred, y)
        # back-prop and optimization step
        loss.backward(retain_graph=True)
        self.predictor_optimizer.step()
        return y_pred, loss

    def step(self):
        self.lfs_optimizer.zero_grad()
        loss, y_pred = self._backward_step()
        loss.backward(retain_graph=True)
        self.lfs_optimizer.step()
        return loss, y_pred

    def _backward_step(self):
        y_pred = self.predictor(self.lossfunc.arch_weights().unsqueeze(0))
        if self.predictor.num_obj > 1:
            target = torch.ones_like(y_pred)
            target[:,1] = 0
        else:
            target = torch.zeros_like(y_pred)
        loss = self.lfs_criterion(y_pred, target)
        return loss, y_pred