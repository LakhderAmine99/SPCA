"""MAGNN ordered instance encoders and destination-normalized attention."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax, scatter
from ..common import SemanticAttention


class InstanceEncoder(nn.Module):
    def __init__(self, channels, heads, encoder='RotatE0'):
        super().__init__()
        self.channels, self.heads, self.encoder = channels, heads, encoder
        if encoder in {'gru', 'lstm'}:
            self.rnn = (nn.GRU if encoder == 'gru' else nn.LSTM)(channels, channels * heads, batch_first=True)
        elif encoder == 'linear':
            self.linear = nn.Linear(channels, channels * heads)
        elif encoder not in {'average', 'RotatE0'}:
            raise ValueError('instance_encoder must be RotatE0, average, linear, gru or lstm')
        if encoder == 'RotatE0' and channels % 2:
            raise ValueError('RotatE0 needs an even hidden width')

    def forward(self, sequence, rotations=None):
        if self.encoder in {'gru', 'lstm'}:
            hidden = self.rnn(sequence)[1]
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            return hidden[-1].reshape(-1, self.heads, self.channels)
        if self.encoder == 'RotatE0':
            values = torch.view_as_complex(sequence.reshape(*sequence.shape[:-1], -1, 2).contiguous())
            rotation = torch.ones_like(rotations[0])
            rotated = [values[:, -1]]
            for i in range(sequence.size(1) - 2, -1, -1):
                rotation = rotation * rotations[i]
                rotated.append(values[:, i] * rotation)
            sequence = torch.view_as_real(torch.stack(rotated, dim=1)).flatten(-2)
        hidden = sequence.mean(1)
        if self.encoder == 'linear':
            return self.linear(hidden).reshape(-1, self.heads, self.channels)
        return hidden[:, None, :].expand(-1, self.heads, -1)


class InstanceAttention(nn.Module):
    def __init__(self, channels, heads, dropout=0.5):
        super().__init__()
        self.attention = nn.Parameter(torch.empty(heads, channels))
        nn.init.xavier_normal_(self.attention, gain=1.414)
        self.dropout = dropout

    def forward(self, instances, destination, num_nodes):
        score = F.leaky_relu((instances * self.attention).sum(-1), 0.01)
        weight = F.dropout(softmax(score, destination, num_nodes=num_nodes), self.dropout, self.training)
        return scatter(instances * weight.unsqueeze(-1), destination, dim=0,
                       dim_size=num_nodes, reduce='sum').flatten(1)
