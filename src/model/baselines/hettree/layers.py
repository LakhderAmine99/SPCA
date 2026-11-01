"""Paper-based HetTree for the shared single-label classification experiments.

Reference: Guan et al., Heterogeneous Graph Neural Network on Semantic Tree,
AAAI 2025, Sections 4.1--4.3, equations (1)--(6).
https://doi.org/10.1609/aaai.v39i16.33860

The paper's microsoft/HetTree repository was unavailable when this module was
written. This is an independent implementation, not a port of the authors' code.
It uses relation-wise mean propagation and PyTorch only. Training labels are
matched to their feature paths after removing all return-walk contributions of
each destination node's own label. No held-out labels or pseudo-labels are used.
"""


import torch


from torch import nn


from torch.nn import functional as F


from ..sehgnn import precompute_metapath_features


def _mlp(in_channels, hidden_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels), nn.PReLU(), nn.Dropout(dropout),
        nn.Linear(hidden_channels, out_channels),
    )


class SubtreeAttention(nn.Module):
    """Equations (4)--(5): parent/children reference and bottom-up attention."""

    def __init__(self, channels, dropout=0.5):
        super().__init__()
        self.reference = _mlp(2 * channels, channels, channels, dropout)
        self.score = nn.Linear(2 * channels, 1, bias=False)

    def forward(self, parent, child_features, child_encodings):
        # The reference uses transformed features M; attention values use the
        # already encoded child subtrees Z, as specified in the paper.
        reference = self.reference(torch.cat([parent, child_features.sum(dim=1)], dim=-1))
        reference = reference.unsqueeze(1).expand(-1, child_encodings.shape[1], -1)
        scores = F.leaky_relu(self.score(torch.cat([reference, child_encodings], dim=-1)), 0.2)
        attention = scores.softmax(dim=1)
        encoded = parent + F.elu((attention * child_encodings).sum(dim=1))
        return encoded, attention


