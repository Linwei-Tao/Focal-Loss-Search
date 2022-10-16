import torch
from torch import nn

class Predictor(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                num_layers=1)
        self.logits_cell = nn.Linear(in_features=self.hidden_size,
                                     out_features=1)

    def forward(self, x, hidden=None):
        out, (hidden, cell) = self.rnn_cell(x, hidden)
        out = self.logits_cell(hidden)
        out = torch.sigmoid(out) * 2
        return out.view(-1)
