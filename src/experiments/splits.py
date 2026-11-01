"""Separate main-pool and low-label protocols, plus leakage-free LP splits."""
import numpy as np
import torch


def node_masks(labels, original, train_ratio, seed, protocol='main'):
    """Keep loader holdouts fixed; vary class-stratified training selection only.

    Main fractions refer to the original training pool per class. Low-label
    fractions refer to all labeled nodes per class, excluding fixed holdouts.
    """
    y = labels.cpu()
    base = {k: original[k].cpu().clone() for k in ('train_mask', 'val_mask', 'test_mask')}
    if any(v.dtype != torch.bool or v.shape != y.shape for v in base.values()):
        raise ValueError('Split masks must be boolean vectors matching labels')
    if any((base[a] & base[b]).any() for a, b in
           [('train_mask', 'val_mask'), ('train_mask', 'test_mask'), ('val_mask', 'test_mask')]):
        raise ValueError('Training, validation and test sets overlap')
    if protocol not in {'main', 'low-label'}:
        raise ValueError('protocol must be main or low-label')
    pool = base['train_mask'] if protocol == 'main' else ~(base['val_mask'] | base['test_mask'])
    pool &= y >= 0
    selected = torch.zeros_like(pool)
    generator = torch.Generator().manual_seed(seed)
    for label in y[y >= 0].unique(sorted=True):
        ids = ((y == label) & pool).nonzero().flatten()
        population = len(ids) if protocol == 'main' else int((y == label).sum())
        count = len(ids) if train_ratio == 'full' else max(1, round(population * train_ratio))
        if count > len(ids) or not len(ids):
            raise ValueError(f'Training budget does not fit class {label.item()} outside holdouts')
        selected[ids[torch.randperm(len(ids), generator=generator)[:count]]] = True
    base['train_mask'] = selected
    if any(not v.any() for v in base.values()):
        raise ValueError('All three splits must be nonempty')
    return {k: v.to(labels.device) for k, v in base.items()}


def negative_pairs(positive_edges, num_source, num_destination, count, seed):
    """Unique negatives excluding *all* observed interactions, not just training."""
    positives = set((positive_edges[0].cpu() * num_destination + positive_edges[1].cpu()).tolist())
    total = num_source * num_destination
    if count > total - len(positives):
        raise ValueError('Not enough unobserved pairs for the requested unique negatives')
    rng = np.random.default_rng(seed)
    chosen, order = set(), []
    while len(order) < count:
        for candidate in rng.integers(total, size=min(max(2 * (count - len(order)), 32), 1_000_000)):
            value = int(candidate)
            if value not in positives and value not in chosen:
                chosen.add(value)
                order.append(value)
                if len(order) == count:
                    break
    flat = torch.tensor(order, dtype=torch.long)
    return torch.stack([flat // num_destination, flat % num_destination])


def link_split(data, relation, ratios, seed, negatives_per_positive=1):
    """Remove held-out interactions in both directions before any preprocessing."""
    from torch_geometric.utils import coalesce
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-8 or min(ratios) <= 0:
        raise ValueError('Three positive LP ratios must sum to one')
    if negatives_per_positive < 1 or not isinstance(negatives_per_positive, int):
        raise ValueError('negatives_per_positive must be a positive integer')
    source, _, destination = relation
    positives = coalesce(data[relation].edge_index.cpu())
    generator = torch.Generator().manual_seed(seed)
    positives = positives[:, torch.randperm(positives.size(1), generator=generator)]
    train_count, val_count = (int(positives.size(1) * r) for r in ratios[:2])
    counts = [train_count, val_count, positives.size(1) - train_count - val_count]
    if min(counts) < 1:
        raise ValueError('Each link split needs a positive edge')
    negatives = negative_pairs(positives, data[source].num_nodes, data[destination].num_nodes,
                              positives.size(1) * negatives_per_positive, seed + 1)
    graph = data.clone()
    training = positives[:, :train_count]
    for edge_type in list(graph.edge_types):
        if edge_type[0] == source and edge_type[2] == destination:
            graph[edge_type].edge_index = training.clone()
        elif edge_type[0] == destination and edge_type[2] == source:
            graph[edge_type].edge_index = training.flip(0)
    splits, offset = {}, 0
    for name, count in zip(('train', 'val', 'test'), counts):
        positive = positives[:, offset:offset + count]
        negative = negatives[:, offset * negatives_per_positive:(offset + count) * negatives_per_positive]
        splits[name] = {'edge_label_index': torch.cat([positive, negative], dim=1),
                        'edge_label': torch.cat([torch.ones(count), torch.zeros(negative.size(1))])}
        offset += count
    return graph, splits
