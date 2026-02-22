import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
################################################################################

class KPLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_layers, output_dim=1, dropout=0, bidirectional=False):

        super(KPLSTM, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.layer1 = nn.LSTM(self.input_dim, self.num_hidden, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional, dtype=torch.float64)

        d = 2 if self.bidirectional else 1
        # Output layer
        self.outlayer = nn.Sequential(
            nn.Linear(d * self.num_hidden, (d * self.num_hidden) // 2, dtype=torch.float64),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear((d * self.num_hidden) // 2, output_dim, dtype=torch.float64)
        )
    #
    def forward(self, x):
        #
        x = torch.permute(x, (1,0,2,3))  # from [batch, seq, kp, xy] to [seq, batch, kp, xy]
        x = torch.flatten(x, start_dim=2)  # [seq, batch, kp*xy]
        seq_length, batch_size, n_keypoints = x.size()

        d = 2 if self.bidirectional else 1
        h0 = torch.zeros([d * self.num_layers, batch_size, self.num_hidden], dtype=torch.float64, device=x.device)
        c0 = torch.zeros([d * self.num_layers, batch_size, self.num_hidden], dtype=torch.float64, device=x.device)

        output, (hn, cn) = self.layer1(x, (h0, c0))

        logits = self.outlayer(output[-1])
        logits = torch.squeeze(logits, dim=1)

        return logits
