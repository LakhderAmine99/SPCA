"""Shared semantic attention and cross-view contrastive objective."""
import torch
from torch import nn
from torch.nn import functional as F


class SemanticAttention(nn.Module):
    def __init__(self, channels, attention_channels=128, dropout=0.0):
        super().__init__()
        self.projection = nn.Linear(channels, attention_channels)
        self.query = nn.Parameter(torch.empty(attention_channels))
        self.dropout = dropout
        nn.init.normal_(self.query, std=0.1)

    def forward(self, views):
        values = torch.stack(views, dim=1)
        query = F.dropout(self.query, self.dropout, self.training)
        weights = (torch.tanh(self.projection(values)).mean(0) @ query).softmax(0)
        return (values * weights[None, :, None]).sum(1)


class CrossViewContrast(nn.Module):
    """HeCo/GTC multi-positive InfoNCE, computed in row blocks for memory."""
    def __init__(self, channels, temperature=0.8, balance=0.5, batch_size=512):
        super().__init__()
        if temperature <= 0 or not 0 <= balance <= 1 or batch_size < 1:
            raise ValueError('Invalid contrastive settings')
        self.project = nn.Sequential(nn.Linear(channels, channels), nn.ELU(), nn.Linear(channels, channels))
        self.temperature, self.balance, self.batch_size = temperature, balance, batch_size

    def forward(self, first, second, positive_edges):
        a, b = (F.normalize(self.project(z), dim=-1) for z in (first, second))
        def direction(source, target):
            losses = []
            for start in range(0, len(source), self.batch_size):
                stop = min(start + self.batch_size, len(source))
                scores = source[start:stop] @ target.T / self.temperature
                selected = (positive_edges[0] >= start) & (positive_edges[0] < stop)
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[positive_edges[0, selected] - start, positive_edges[1, selected]] = True
                mask[torch.arange(stop - start, device=source.device), torch.arange(start, stop, device=source.device)] = True
                losses.append((scores.logsumexp(1) - scores.masked_fill(~mask, -torch.inf).logsumexp(1)).sum())
            return torch.stack(losses).sum() / len(source)
        return self.balance * direction(a, b) + (1 - self.balance) * direction(b, a)
