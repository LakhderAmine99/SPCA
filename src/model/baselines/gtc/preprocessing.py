"""Hop2Token: root features and normalized repeated metapath propagation."""
import torch
from torch_geometric.utils import add_remaining_self_loops


@torch.no_grad()
def hop_tokens(x, graphs, hops=2):
    if hops < 1:
        raise ValueError('Hop2Token requires at least one hop')
    output = []
    for graph in graphs:
        graph, _ = add_remaining_self_loops(graph, num_nodes=len(x))
        sequence = [x.detach()]
        degree = torch.bincount(graph[1], minlength=len(x)).clamp_min(1)
        adjacency = torch.sparse_coo_tensor(graph.flip(0), 1 / degree[graph[1]].to(x.dtype),
                                             (len(x), len(x)), check_invariants=True).coalesce()
        for _ in range(hops):
            sequence.append(torch.sparse.mm(adjacency, sequence[-1]))
        output.append(torch.stack(sequence, dim=1))
    return output
