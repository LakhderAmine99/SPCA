"""Fixed held-out sets and nested training samples for label-budget experiments."""

import torch
from sklearn.model_selection import train_test_split


def _stratified_order(labels, indices, seed):
    """Order a training pool so growing prefixes retain roughly its class mix."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.as_tensor(indices, dtype=torch.long)
    groups = []
    for label in labels[indices].unique(sorted=True):
        group = indices[labels[indices] == label]
        groups.append(group[torch.randperm(group.numel(), generator=generator)].tolist())

    # Every supported training budget contains at least one node of each class.
    class_order = torch.randperm(len(groups), generator=generator).tolist()
    order = [groups[class_id][0] for class_id in class_order]
    drawn = [1] * len(groups)
    sizes = [len(group) for group in groups]
    proportions = [size / len(indices) for size in sizes]
    while len(order) < len(indices):
        target_size = len(order) + 1
        # Add a node from the class furthest below its desired proportion.
        class_id = max(
            (i for i in range(len(groups)) if drawn[i] < sizes[i]),
            key=lambda i: target_size * proportions[i] - drawn[i],
        )
        order.append(groups[class_id][drawn[class_id]])
        drawn[class_id] += 1
    return torch.tensor(order, dtype=torch.long)


def make_fixed_masks(labels, train_ratio, val_count, seed=42, test_ratio=0.10, test_count=None):
    """Reserve validation/test first; train on a prefix of one stratified pool.

    Numeric training and test ratios are fractions of all target nodes. An
    explicit test_count overrides test_ratio with an exact node count. The
    training budget 'full' uses every node outside validation and test. With a fixed
    seed and holdout configuration, validation/test membership is independent of
    the requested training ratio, and smaller training sets are always subsets
    of larger ones. Unselected training-pool labels remain unused.
    """
    use_full_pool = train_ratio == 'full'
    if not use_full_pool and not 0 < train_ratio <= 1:
        raise ValueError('train_ratio must be greater than 0 and at most 1')
    if test_count is not None and (type(test_count) is not int or test_count < 1):
        raise ValueError('test_count must be a positive integer')
    if test_count is None and not 0 < test_ratio < 1:
        raise ValueError('test_ratio must be greater than 0 and less than 1')
    cpu_labels = labels.detach().cpu()
    num_nodes = cpu_labels.numel()
    num_classes = cpu_labels.unique().numel()
    test_count = round(num_nodes * test_ratio) if test_count is None else test_count
    pool_count = num_nodes - val_count - test_count
    train_count = pool_count if use_full_pool else round(num_nodes * train_ratio)
    if num_classes == 0:
        raise ValueError('A fixed split needs labeled nodes')
    if train_count < num_classes:
        raise ValueError('Training budget must select at least one node per class')
    if val_count < num_classes or test_count < num_classes or pool_count < num_classes:
        raise ValueError('Fixed validation, test, and training pools each need at least one node per class')
    if train_count > pool_count:
        raise ValueError(
            f'train_ratio selects {train_count} nodes, but the fixed training pool '
            f'has {pool_count} ({pool_count / num_nodes:.2%} of all nodes); '
            f'leave room for {val_count} validation and {test_count} test nodes. '
            'Reduce train_ratio or the test size (--test-ratio/--test-count).'
        )

    all_indices = torch.arange(num_nodes).numpy()
    remaining, val_indices = train_test_split(
        all_indices, test_size=val_count, random_state=seed,
        stratify=cpu_labels.numpy(),
    )
    pool_indices, test_indices = train_test_split(
        remaining, test_size=test_count, random_state=seed,
        stratify=cpu_labels[remaining].numpy(),
    )
    order = _stratified_order(cpu_labels, pool_indices, seed)
    masks = {}
    for name, indices in [('train_mask', order[:train_count]),
                          ('val_mask', val_indices), ('test_mask', test_indices)]:
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=labels.device)
        mask[torch.as_tensor(indices, dtype=torch.long, device=labels.device)] = True
        masks[name] = mask
    return masks
