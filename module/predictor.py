import torch
from torch import nn

class Predictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_obj=1, predictor_lambda=0):
        # if mo=1, means only one objective
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_objective = num_obj


        self.embedding = nn.Linear(in_features=self.input_size,
                                   out_features=self.hidden_size)

        self.rnn_cell = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                bidirectional=False,
                                num_layers=1)

        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=num_obj)

        self.predictor_lambda = predictor_lambda
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, (hidden, cell) = self.rnn_cell(x, hidden)
        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        return out.reshape(-1, self.num_objective)


