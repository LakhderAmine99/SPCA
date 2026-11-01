"""Low-rank fusion of features, node types and incident relations."""
import torch
from torch import nn


class BlendFusion(nn.Module):
    def __init__(self, in_channels, type_channels=512, out_channels=64, rank=4):
        super().__init__()
        self.factors = nn.ParameterList([nn.Parameter(torch.empty(rank, width + 1, out_channels))
                                        for width in (in_channels, type_channels, type_channels)])
        self.weights = nn.Parameter(torch.empty(rank))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        for factor in self.factors:
            nn.init.xavier_uniform_(factor)
        nn.init.normal_(self.weights, std=rank ** -0.5)

    def forward(self, features, node_types, relations):
        parts = [torch.einsum('ni,ric->nrc', torch.cat([x, x.new_ones((len(x), 1))], -1), factor)
                 for x, factor in zip((features, node_types, relations), self.factors)]
        return ((parts[0] * parts[1] * parts[2]) * self.weights[None, :, None]).sum(1).add(self.bias).relu()
