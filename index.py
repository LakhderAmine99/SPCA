import argparse
import csv
import statistics
import time
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from src.checkpoint import save_checkpoint
from src.imdb import IMDB
from src.acm import ACM
from src.dblp import DBLP
from src.datasets import DATASET_INFO, metapath_configuration, prepare_benchmark
from src.splits import make_fixed_masks
from src.model.baselines.sehgnn import SeHGNN, precompute_metapath_features
from src.model.baselines.simplehgn import SimpleHGN
from src.model.baselines.hettree import HetTree, precompute_hettree_inputs
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.model.spca import (
    SimpleMetapathConv,
    SubpathHANConv,
    SubpathHANConv2,
    RegularizedSubpathHANConv,
    SubpathHANConv3,
    StackedSubpathHAN,
    SubpathClassifier,
)


def train(model, data, optimizer, criterion, device):
    target = data[getattr(data, 'target_type', 'movie')]
    mask = target.train_mask
    if torch.any(mask & (target.val_mask | target.test_mask)):
        raise ValueError('Training mask overlaps validation or test labels')
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[mask], target.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, device, splits=None):
    target = data[getattr(data, 'target_type', 'movie')]
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=-1)

    results = {}
    for split in (['train_mask', 'val_mask', 'test_mask'] if splits is None else splits):
        mask = target[split]
        y_true = target.y[mask].cpu()
        y_pred = preds[mask].cpu()

        acc = (y_pred == y_true).float().mean().item()
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        loss = F.cross_entropy(out[mask], target.y[mask]).item()
        results[split] = {'loss': loss, 'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    return results


def parse_train_ratio(value):
    """Accept a fraction such as 0.10 or a percentage such as 10%."""
    value = value.strip()
    try:
        ratio = float(value[:-1]) / 100 if value.endswith('%') else float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            'Use a fraction such as 0.10 or a percentage such as 10%'
        ) from None
    if not 0 < ratio <= 1:
        raise argparse.ArgumentTypeError(
            'Training ratio must be greater than 0 and at most 1 (100%)'
        )
    return ratio


def parse_training_budget(value):
    """Accept a numeric share or 'full' for all non-held-out training labels."""
    if value.strip().lower() == 'full':
        return 'full'
    return parse_train_ratio(value)


def make_train_mask(labels, original_train_mask, train_ratio, seed=42):
    """Use a fraction of target nodes, drawing only from the original train split."""
    if not 0 < train_ratio <= 1:
        raise ValueError('train_ratio must be greater than 0 and at most 1')

    num_train = round(labels.numel() * train_ratio)
    train_indices = original_train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    if num_train < 1:
        raise ValueError('train_ratio selects fewer than one target node')
    if num_train > len(train_indices):
        raise ValueError(
            f'Requested {num_train} training nodes, but the original train split '
            f'has only {len(train_indices)}. A larger ratio requires a new '
            'train/validation/test split.'
        )
    if num_train == len(train_indices):
        return original_train_mask.clone()

    # Stratification keeps the genre proportions close to the original split.
    selected, _ = train_test_split(
        train_indices, train_size=num_train, random_state=seed,
        stratify=labels[original_train_mask].detach().cpu().numpy(),
    )
    train_mask = torch.zeros_like(original_train_mask)
    train_mask[torch.as_tensor(selected, device=train_mask.device)] = True
    return train_mask


def make_imdb_masks(
    labels, train_mask, val_mask, test_mask, train_ratio, seed=42,
    split_protocol='fixed', test_ratio=0.10, test_count=None,
):
    """Use fixed holdouts by default; retain the previous splits as legacy mode."""
    if split_protocol not in {'fixed', 'legacy'}:
        raise ValueError('split_protocol must be fixed or legacy')
    if test_count is not None and split_protocol != 'fixed':
        raise ValueError('test_count requires split_protocol=\'fixed\'')
    if split_protocol == 'fixed':
        return make_fixed_masks(
            labels, train_ratio, int(val_mask.sum()), seed=seed, test_ratio=test_ratio,
            test_count=test_count,
        ), 'fixed-nested'

    if train_ratio == 'full':
        raise ValueError("train_ratio='full' requires split_protocol='fixed'")
    if not 0 < train_ratio <= 1:
        raise ValueError('train_ratio must be greater than 0 and at most 1')

    num_nodes = labels.numel()
    num_train = round(num_nodes * train_ratio)
    if num_train <= int(train_mask.sum()):
        return {
            'train_mask': make_train_mask(labels, train_mask, train_ratio, seed),
            'val_mask': val_mask.clone(), 'test_mask': test_mask.clone(),
        }, 'original'

    num_val = int(val_mask.sum())
    num_classes = labels.unique().numel()
    if num_val < num_classes:
        raise ValueError('A stratified split needs at least one validation node per class')
    if num_nodes - num_train - num_val < num_classes:
        raise ValueError(
            f'train_ratio selects {num_train} nodes; a stratified split must '
            f'leave room for {num_val} validation nodes and at least '
            f'{num_classes} test nodes'
        )

    all_labels = labels.detach().cpu().numpy()
    train_indices, held_out = train_test_split(
        torch.arange(num_nodes).numpy(), train_size=num_train,
        random_state=seed, stratify=all_labels,
    )
    val_indices, test_indices = train_test_split(
        held_out, train_size=num_val, random_state=seed,
        stratify=all_labels[held_out],
    )
    masks = {}
    for name, indices in [('train_mask', train_indices), ('val_mask', val_indices),
                          ('test_mask', test_indices)]:
        mask = torch.zeros_like(train_mask)
        mask[torch.as_tensor(indices, device=mask.device)] = True
        masks[name] = mask
    return masks, 'stratified'


def fit_imdb_model(
    model, name, data, device, checkpoint_path, epochs=249, patience=100,
    learning_rate=0.005, weight_decay=5e-4, selection_metric='loss',
):
    """Shared optimization, validation selection and evaluation for all models."""
    if selection_metric not in {'loss', 'f1_macro'}:
        raise ValueError('selection_metric must be loss or f1_macro')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_score = -float('inf')
    best_epoch = 0
    patience_counter = 0
    epoch_times = []
    training_times = []
    def synchronize():
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize(device)
    print(f'\nTraining {name}')
    for epoch in range(1, epochs + 1):
        synchronize()
        epoch_start = time.perf_counter()
        loss = train(model, data, optimizer, criterion, device)
        synchronize()
        training_times.append(time.perf_counter() - epoch_start)
        # Test metrics are computed once after validation selects a checkpoint.
        results = test(model, data, device, ('train_mask', 'val_mask'))
        synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)
        val_loss = results['val_mask']['loss']
        print(f'{name} | Epoch {epoch:02d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
              f"Train Acc: {results['train_mask']['acc']:.4f} | Val Acc: {results['val_mask']['acc']:.4f} | "
              f"Val F1-micro: {results['val_mask']['f1_micro']:.4f} | "
              f"Val F1-macro: {results['val_mask']['f1_macro']:.4f}")
        score = -val_loss if selection_metric == 'loss' else results['val_mask']['f1_macro']
        if math.isfinite(val_loss) and (score > best_score or (score == best_score and val_loss < best_val_loss)):
            best_score = score
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            saved_path = save_checkpoint(model.state_dict(), checkpoint_path)
            if saved_path != checkpoint_path:
                print(f'Checkpoint replacement was locked; saved best model to {saved_path}')
            checkpoint_path = saved_path
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'{name}: early stopping at epoch {epoch}')
                break

    if best_epoch == 0:
        raise RuntimeError(f'{name} did not produce a finite validation loss')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    results = test(model, data, device)
    print(f'{name} best epoch: {best_epoch} | Checkpoint: {checkpoint_path}')
    print(f"{name} Final Test Results: {results['test_mask']}")
    return {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'selection_metric': selection_metric,
        'checkpoint': str(checkpoint_path), 'metrics': results,
        'epochs_run': len(epoch_times),
        'epoch_times_seconds': epoch_times,
        'epoch_time_mean_seconds': statistics.mean(epoch_times),
        'epoch_time_std_seconds': statistics.pstdev(epoch_times),
        'epoch_time_median_seconds': statistics.median(epoch_times),
        'epoch_time_min_seconds': min(epoch_times),
        'epoch_time_max_seconds': max(epoch_times),
        'training_time_mean_seconds': statistics.mean(training_times),
        'total_epoch_time_seconds': sum(epoch_times),
        'timing_scope': 'training step plus train/validation evaluation; excludes checkpoint I/O, logging, preprocessing and final test; all epochs included',
        'device': str(device),
    }


def imdb_metapath_configuration():
    """Shared path definitions for normal runs and validation-only tuning."""
    return metapath_configuration('imdb')


def run_experiment(
    train_ratio=0.01, split_seed=42, model_name='both', epochs=249,
    patience=100, checkpoint_dir=None, split_protocol='fixed', test_ratio=0.10,
    subpath_variant='original', learning_rate=None, weight_decay=5e-4,
    dropout=None, selection_metric='loss', dataset_name='imdb', data_root=None,
    hettree_hops=None, hettree_use_labels=True, test_count=None,
    subpath_encoder='concat-mlp', shared_subpath_initialization=False,
):
    if subpath_encoder not in {'concat-mlp', 'mean', 'lstm', 'transformer'}:
        raise ValueError('subpath_encoder must be concat-mlp, mean, lstm, or transformer')
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_INFO:
        raise ValueError('dataset_name must be imdb, dblp, or acm')
    if model_name not in {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias', 'sehgnn', 'simplehgn', 'hettree', 'both', 'ablation', 'all'}:
        raise ValueError(
            'model_name must be subpath, no-context, metapath-bias, metapath-scalar-bias, '
            'sehgnn, simplehgn, hettree, both, ablation, or all'
        )
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    metapaths, subpaths_dict = metapath_configuration(dataset_name)
    hettree_depth = max(len(path) - 1 for path in metapaths)
    hettree_hops = hettree_depth if hettree_hops is None else hettree_hops
    if not isinstance(hettree_hops, int) or hettree_hops < 1:
        raise ValueError('hettree_hops must be a positive integer')
    if model_name in {'hettree', 'all'} and hettree_hops < hettree_depth:
        raise ValueError(f'hettree_hops must be at least {hettree_depth} to retain all shared {dataset_name.upper()} metapaths')
    if subpath_variant not in {'original', 'regularized'}:
        raise ValueError('subpath_variant must be original or regularized')
    if selection_metric not in {'loss', 'f1_macro'}:
        raise ValueError('selection_metric must be loss or f1_macro')
    learning_rate = (0.001 if subpath_variant == 'regularized' else 0.005) if learning_rate is None else learning_rate
    dropout = (0.5 if subpath_variant == 'regularized' else 0.0) if dropout is None else dropout
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('weight_decay must be finite and non-negative')
    if not 0 <= dropout < 1:
        raise ValueError('dropout must be at least zero and less than one')
    if split_protocol not in {'fixed', 'legacy'}:
        raise ValueError('split_protocol must be fixed or legacy')
    if train_ratio == 'full' and split_protocol != 'fixed':
        raise ValueError("train_ratio='full' requires split_protocol='fixed'")
    if test_count is not None:
        if split_protocol != 'fixed':
            raise ValueError('test_count requires split_protocol=\'fixed\'')
        if type(test_count) is not int or test_count < 1:
            raise ValueError('test_count must be a positive integer')
    if split_protocol == 'fixed' and test_count is None and not 0 < test_ratio < 1:
        raise ValueError('test_ratio must be greater than 0 and less than 1')
    if data_root is None:
        project_root = Path(__file__).resolve().parent
        data_root = project_root / 'data' / 'hgb' / dataset_name
        old_dblp_root = project_root / 'data' / 'dblp'
        if dataset_name == 'dblp' and not data_root.exists() and old_dblp_root.exists():
            data_root = old_dblp_root
    data_root = Path(data_root).expanduser().resolve()
    dataset = {'imdb': IMDB, 'dblp': DBLP, 'acm': ACM}[dataset_name](root=str(data_root))
    data = prepare_benchmark(dataset[0], dataset_name)
    target_type = DATASET_INFO[dataset_name][0]
    print(f'Dataset: {dataset_name.upper()} | Target: {target_type} | Cache: {data_root}')
    if hasattr(data, 'feature_source'):
        print(f'Features: {data.feature_source}')

    masks, split_strategy = make_imdb_masks(
        data[target_type].y, data[target_type].train_mask, data[target_type].val_mask,
        data[target_type].test_mask, train_ratio, seed=split_seed,
        split_protocol=split_protocol, test_ratio=test_ratio, test_count=test_count,
    )
    for name, mask in masks.items():
        data[target_type][name] = mask

    if split_strategy == 'fixed-nested':
        held_out_count = int(masks['test_mask'].sum())
        held_out_share = held_out_count / data[target_type].y.numel()
        print('Split: fixed stratified validation/test sets; nested training subsets '
              f'(test: {held_out_count} nodes, {held_out_share:.2%}; seed: {split_seed})')
    elif split_strategy == 'stratified':
        print('Split: legacy stratified split; validation/test membership changes '
              'with the training ratio')
    else:
        print('Split: sampled original training pool; original validation/test sets')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    num_nodes = data[target_type].y.numel()

    num_train = int(data[target_type].train_mask.sum())
    num_val = int(data[target_type].val_mask.sum())
    num_test = int(data[target_type].test_mask.sum())
    print(f'Training: {num_train}/{num_nodes} {target_type} nodes ({num_train / num_nodes:.2%})')
    if train_ratio == 'full':
        print('Training budget: full pool; every label outside validation/test is used')
    print(f'Validation: {num_val} | Test: {num_test} | '
          f'Unused labels: {num_nodes - num_train - num_val - num_test}')

    feature_hops = hettree_depth
    print('Shared metapaths for all models:', [' -> '.join(path) for path in metapaths])

    in_channels_dict = {k: v.size(-1) for k, v in data.x_dict.items()}
    num_classes = int(data[target_type].y.max()) + 1
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else (
        Path(__file__).resolve().parent / 'data' / 'checkpoints'
    )
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    budget_tag = 'full' if train_ratio == 'full' else f'{train_ratio * 100:g}pct'
    run_tag = f'{budget_tag}_seed{split_seed}'
    if split_protocol == 'fixed':
        # Keep results from this protocol separate from previous experiments.
        test_tag = f'{test_count}nodes' if test_count is not None else f'{test_ratio * 100:g}pct'
        run_tag = f'fixed_test{test_tag}_{run_tag}'
    if getattr(data, 'preprocessing', 'original') == 'raw-title-v1':
        run_tag = f'raw_title_{run_tag}'
    if model_name in {'simplehgn', 'all'}:
        run_tag = f'metagraph_{run_tag}'
    if model_name in {'hettree', 'all'}:
        run_tag = f"sharedpaths_tree{hettree_depth}h_{'labels' if hettree_use_labels else 'features'}_{run_tag}"
    # Isolate this protocol from earlier runs with model-specific path inputs.
    run_tag = f'sharedpaths_v2_{run_tag}'
    if subpath_variant != 'original' or dropout != 0.0 or learning_rate != 0.005 or weight_decay != 5e-4 or selection_metric != 'loss':
        configuration = {
            'variant': subpath_variant, 'learning_rate': learning_rate,
            'weight_decay': weight_decay, 'dropout': dropout, 'selection_metric': selection_metric,
        }
        config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:8]
        run_tag += f"_{'reg' if subpath_variant == 'regularized' else 'tune'}{config_hash}"
    if subpath_encoder != 'concat-mlp':
        run_tag += f'_encoder_{subpath_encoder}'
    if subpath_encoder in {'lstm', 'transformer'} or shared_subpath_initialization:
        run_tag += '_persubpath_v2'
    if shared_subpath_initialization:
        run_tag += '_sharedinit'
    groups = {
        'both': ['subpath', 'sehgnn'],
        'ablation': ['subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'],
        'all': ['subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias', 'sehgnn', 'simplehgn', 'hettree'],
    }
    names = groups.get(model_name, [model_name])
    comparison = {}
    for name in names:
        # Give each model a reproducible initialization independent of run order.
        torch.manual_seed(split_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(split_seed)
        if name in {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'}:
            encoder_options = dict(
                in_channels_dict=in_channels_dict, out_channels=128,
                metadata=data.metadata(), metapaths=metapaths,
                subpaths_dict=subpaths_dict, heads=8, aggregator=subpath_encoder,
            )
            encoder_options['dropout'] = dropout
            encoder_class = RegularizedSubpathHANConv if subpath_variant == 'regularized' else SubpathHANConv2
            encoder = encoder_class(**encoder_options)
            model = SubpathClassifier(encoder, 128, num_classes)
            if shared_subpath_initialization:
                # A fixed reference makes common weights independent of encoder RNG use.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(split_seed)
                    reference_options = dict(encoder_options, aggregator='concat-mlp')
                    reference = SubpathClassifier(encoder_class(**reference_options), 128, num_classes)
                context_prefixes = ('encoder.subpath_mlps.', 'encoder.subpath_sequence_encoders.',
                                    'encoder.subpath_positions', 'encoder.metapath_bias.')
                state = model.state_dict()
                for key, value in reference.state_dict().items():
                    if key in state and not key.startswith(context_prefixes):
                        state[key] = value
                model.load_state_dict(state)
            if name != 'subpath':
                # Reuse exactly the baseline's shared initial weights, including
                # the classifier. Context construction must not shift its RNG.
                with torch.random.fork_rng(devices=[]):
                    ablation_encoder = encoder_class(
                        **encoder_options, context_mode=name,
                    )
                for module_name in ('proj', 'final_lin', 'norm', 'hop_norm', 'output_norm'):
                    if not hasattr(encoder, module_name):
                        continue
                    getattr(ablation_encoder, module_name).load_state_dict(
                        getattr(encoder, module_name).state_dict()
                    )
                model.encoder = ablation_encoder
                del encoder
            model = model.to(device)
        elif name == 'sehgnn':
            print(f'\nPrecomputing SeHGNN features (shared metapaths; {feature_hops} hops; no labels)')
            features = precompute_metapath_features(
                data.x_dict, data.edge_index_dict, target_type=target_type, num_hops=feature_hops,
                metapaths=metapaths,
            )
            print('SeHGNN feature paths:', [" -> ".join(path) for path in sorted(features)])
            model = SeHGNN(features, num_classes=num_classes, hidden_channels=128).to(device)
        elif name == 'hettree':
            label_mode = 'training labels; self contribution removed' if hettree_use_labels else 'no labels'
            print(f'\nPrecomputing HetTree semantic tree (shared metapaths; {hettree_depth} hops; {label_mode})')
            print('HetTree selected metapaths:', [' -> '.join(path) for path in metapaths])
            features, label_features = precompute_hettree_inputs(
                data.x_dict, data.edge_index_dict, target_type=target_type, num_hops=hettree_depth,
                labels=data[target_type].y, train_mask=data[target_type].train_mask,
                num_classes=num_classes, use_labels=hettree_use_labels, metapaths=metapaths,
                include_ancestor_features=False,
            )
            model = HetTree(features, num_classes=num_classes, label_features=label_features,
                            hidden_channels=128, dropout=0.5).to(device)
            print('HetTree tree paths (shared metapaths and zero-input structural ancestors):',
                  [' -> '.join(path) for path in model.paths])
            print('HetTree label paths:', [' -> '.join(path) for path in model.label_paths])
        else:
            print('\nProjecting Simple-HGN graph along the shared metapaths')
            model = SimpleHGN(data.x_dict, data.edge_index_dict, num_classes=num_classes,
                              target_type=target_type, metapaths=metapaths).to(device)
            print(f'Simple-HGN graph: {model.graph_mode} | Nodes: {model.target_count} target nodes | '
                  f'Edges: {model.edge_index.shape[1]:,}')
        checkpoint_path = checkpoint_dir / f'{dataset_name}_{name}_{run_tag}.pth'
        comparison[name] = fit_imdb_model(
            model, name, data, device, checkpoint_path, epochs=epochs, patience=patience,
            learning_rate=learning_rate, weight_decay=weight_decay, selection_metric=selection_metric,
        )
        comparison[name]['num_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        comparison[name]['selected_metapaths'] = [list(path) for path in metapaths]
        if name == 'simplehgn':
            comparison[name]['graph_mode'] = model.graph_mode
        if name == 'hettree':
            comparison[name].update({
                'implementation': 'paper-equations-v1', 'feature_hops': hettree_depth,
                'path_scope': 'shared-metapaths-only',
                'selected_metapaths': [list(path) for path in metapaths],
                'feature_paths': [list(path) for path in model.paths
                                  if len(path) == 1 or path in metapaths],
                'tree_paths': [list(path) for path in model.paths],
                'tree_ancestor_paths': [list(path) for path in model.paths
                                        if len(path) > 1 and path not in metapaths],
                'ancestor_feature_inputs': 'zeros',
                'label_paths': [list(path) for path in model.label_paths],
                'label_propagation': model.uses_labels, 'tree_node_count': len(model.paths),
            })
        if name in {'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias'}:
            comparison[name]['context_mode'] = model.encoder.context_mode
            comparison[name]['context_injection'] = 'none' if name == 'no-context' else 'add-before-concat-linear'
            if name in {'metapath-bias', 'metapath-scalar-bias'}:
                comparison[name]['bias_dimensions'] = 1 if name == 'metapath-scalar-bias' else 128
            context_prefixes = ('type_emb.', 'subpath_mlps.', 'metapath_bias.',
                                'subpath_sequence_encoders.', 'subpath_positions')
            comparison[name]['context_parameters'] = sum(
                p.numel() for key, p in model.encoder.named_parameters()
                if key.startswith(context_prefixes) and p.requires_grad
            )
            comparison[name]['subpath_encoder_sharing'] = 'per-distinct-subpath'
            comparison[name]['subpath_encoder_count'] = (
                len(model.encoder.subpath_mlps) if name == 'subpath' and subpath_encoder == 'concat-mlp'
                else len(model.encoder.subpath_sequence_encoders)
                if name == 'subpath' and subpath_encoder in {'lstm', 'transformer'} else 0
            )

    print(f'\n{dataset_name.upper()} comparison (checkpoints selected by validation {selection_metric})')
    print(f"{'Model':<22} {'Parameters':>12} {'Best epoch':>10} {'Test Acc':>10} {'F1-micro':>10} {'F1-macro':>10}")
    for name, result in comparison.items():
        metrics = result['metrics']['test_mask']
        print(f"{name:<22} {result['num_parameters']:>12,} {result['best_epoch']:>10} {metrics['acc']:>10.4f} "
              f"{metrics['f1_micro']:>10.4f} {metrics['f1_macro']:>10.4f}")
    report = {
        'dataset': DATASET_INFO[dataset_name][1],
        'dataset_name': dataset_name, 'target_type': target_type, 'num_classes': num_classes,
        'data_root': str(data_root),
        'node_counts': {node_type: data[node_type].num_nodes for node_type in data.node_types},
        'preprocessing': getattr(data, 'preprocessing', 'original'),
        'feature_source': getattr(data, 'feature_source', 'Original dataset features; identity for featureless nodes'),
        'train_ratio': num_train / num_nodes if train_ratio == 'full' else train_ratio,
        'requested_train_ratio': train_ratio,
        'split_strategy': split_strategy,
        'split_protocol': split_protocol,
        'test_ratio': (num_test / num_nodes if test_count is not None else test_ratio)
                      if split_protocol == 'fixed' else None,
        'requested_test_count': test_count,
        'requested_test_ratio': test_ratio if split_protocol == 'fixed' and test_count is None else None,
        'unused_count': num_nodes - num_train - num_val - num_test,
        'train_pool_count': num_nodes - num_val - num_test,
        'split_indices': {
            split: data[target_type][f'{split}_mask'].nonzero(as_tuple=True)[0].cpu().tolist()
            for split in ['train', 'val', 'test']
        },
        'seed': split_seed, 'train_count': num_train, 'val_count': num_val,
        'test_count': num_test, 'max_epochs': epochs, 'patience': patience,
        'learning_rate': learning_rate, 'weight_decay': weight_decay,
        'selection_metric': selection_metric,
        'metapath_protocol': 'sharedpaths-v2', 'shared_metapaths': metapaths,
        'subpath_configuration': {
            'variant': subpath_variant,
            'hidden_channels': 128, 'aggregator': subpath_encoder, 'dropout': dropout,
            'encoder_protocol': 'per-subpath-v2', 'encoder_sharing': 'per-distinct-subpath',
            'attention_heads': 8 if subpath_variant == 'regularized' else 1,
            'feature_l2_normalization': subpath_variant == 'regularized',
            'node_feature_residuals': subpath_variant == 'regularized',
            'metapaths': metapaths,
            'subpaths': {' -> '.join(path): subpaths_dict[path] for path in metapaths},
            'context_injection': 'none' if model_name == 'no-context' else 'add-before-concat-linear',
            'metapath_bias_initialization': 'zeros', 'shared_initialization': True,
            'shared_initialization_across_aggregators': shared_subpath_initialization,
            'bias_dimensions': {'metapath-bias': 128, 'metapath-scalar-bias': 1},
        },
        'sehgnn_configuration': {
            'feature_hops': feature_hops, 'hidden_channels': 128, 'projection_layers': 2,
            'metapaths': metapaths, 'include_raw_features': True,
            'task_layers': 4, 'dropout': 0.5, 'label_propagation': False,
        },
        'simplehgn_configuration': {
            'graph_mode': 'metapath-projected',
            'metapaths': metapaths,
            'hidden_channels_per_head': 64, 'hidden_layers': 2, 'heads': 8,
            'edge_channels': 64, 'dropout': 0.5, 'negative_slope': 0.05,
            'attention_residual': 0.05, 'self_loops': True, 'output_l2_normalization': True,
        },
        'hettree_configuration': {
            'implementation': 'paper-equations-v1',
            'reference': 'https://doi.org/10.1609/aaai.v39i16.33860',
            'feature_hops': hettree_depth, 'max_hops': hettree_hops,
            'path_scope': 'shared-metapaths-only', 'metapaths': metapaths,
            'ancestor_feature_inputs': 'zeros',
            'hidden_channels': 128, 'dropout': 0.5, 'mlp_layers': 2,
            'feature_aggregation': 'relation-wise-mean',
            'label_propagation': hettree_use_labels, 'label_source': 'training-mask-only',
            'self_label_exclusion': 'exact-propagation-diagonal',
            'subtree_reference': 'parent-and-sum-of-child-features',
            'tree_readout': 'sum-of-all-encoded-tree-nodes',
            'feature_residual': True, 'label_residual': hettree_use_labels,
        },
        'models': comparison,
    }
    report_path = checkpoint_dir / f'{dataset_name}_{model_name}_{run_tag}_results.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Comparison results: {report_path}')
    return comparison


def run_aggregator_ablation(seeds=None, **options):
    """Compare all four context aggregators with paired splits and common weights."""
    options = dict(options)
    seeds = [options.get('split_seed', 42)] if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide at least one seed, without duplicates')
    options.pop('model_name', None)
    options.pop('subpath_encoder', None)
    options.pop('shared_subpath_initialization', None)
    directory = Path(options.get('checkpoint_dir') or 'data/checkpoints')
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in options.items() if key != 'split_seed'}
    configuration['encoder_protocol'] = 'per-subpath-v2'
    tag = hashlib.sha256(json.dumps({'options': configuration, 'seeds': seeds}, sort_keys=True).encode()).hexdigest()[:10]
    base = directory / f"{options.get('dataset_name', 'imdb')}_aggregator_ablation_{tag}"
    rows, runs = [], []
    started = time.perf_counter()
    for seed in seeds:
        for aggregator in ('concat-mlp', 'mean', 'lstm', 'transformer'):
            print(f'\nAggregator ablation | Seed {seed} | Encoder {aggregator} | Per-subpath encoders')
            result = run_experiment(
                **dict(options, split_seed=seed, model_name='subpath', subpath_encoder=aggregator,
                       shared_subpath_initialization=True),
            )['subpath']
            metrics = result['metrics']['test_mask']
            row = dict(aggregator=aggregator, seed=seed, test_accuracy=metrics['acc'],
                       test_f1_micro=metrics['f1_micro'], test_f1_macro=metrics['f1_macro'],
                       test_loss=metrics['loss'], num_parameters=result['num_parameters'],
                       context_parameters=result['context_parameters'], best_epoch=result['best_epoch'],
                       best_val_loss=result['best_val_loss'], epochs_run=result['epochs_run'],
                       **{key: result[key] for key in (
                           'epoch_time_mean_seconds', 'epoch_time_std_seconds', 'epoch_time_median_seconds',
                           'epoch_time_min_seconds', 'epoch_time_max_seconds', 'training_time_mean_seconds',
                           'total_epoch_time_seconds', 'device', 'checkpoint')})
            rows.append(row)
            runs.append(dict(aggregator=aggregator, seed=seed, **result))
            # Save completed runs progressively, so a long sweep retains partial results.
            with base.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            base.with_suffix('.json').write_text(json.dumps(
                dict(configuration=configuration, seeds=seeds, complete=False, runs=runs), indent=2), encoding='utf-8')
    summary = []
    for aggregator in ('concat-mlp', 'mean', 'lstm', 'transformer'):
        selected = [row for row in rows if row['aggregator'] == aggregator]
        entry = dict(aggregator=aggregator, seeds=len(selected))
        for metric in ('test_accuracy', 'test_f1_micro', 'test_f1_macro', 'epoch_time_mean_seconds',
                       'training_time_mean_seconds', 'total_epoch_time_seconds', 'epochs_run',
                       'num_parameters', 'context_parameters', 'best_epoch'):
            values = [row[metric] for row in selected]
            entry[metric + '_mean'] = statistics.mean(values)
            entry[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(entry)
    report = dict(configuration=configuration, seeds=seeds, complete=True, runs=runs, summary=summary,
                  wall_time_seconds=time.perf_counter() - started,
                  timing_scope=runs[0]['timing_scope'],
                  summary_scope='Equal-weight means across seeds; sample standard deviations across seeds (null for one seed). Scores are fractions.')
    base.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with Path(str(base) + '_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print('\nAggregator ablation (scores in %, time in seconds; mean ? seed sample std)')
    print(f"{'Aggregator':<14} {'Accuracy':>17} {'F1-micro':>17} {'F1-macro':>17} {'Epoch seconds':>17} {'Parameters':>12}")
    for entry in summary:
        def cell(key, scale=1):
            value = entry[key + '_mean'] * scale
            std = entry[key + '_std']
            return f'{value:.4f}' if std is None else f'{value:.4f} ? {std * scale:.4f}'
        print(f"{entry['aggregator']:<14} {cell('test_accuracy', 100):>17} "
              f"{cell('test_f1_micro', 100):>17} {cell('test_f1_macro', 100):>17} "
              f"{cell('epoch_time_mean_seconds'):>17} {entry['num_parameters_mean']:>12,.0f}")
    print(f'Reports: {base}.json | {base}.csv | {base}_summary.csv')
    return report


# Preserve the notebook and tuning API used before dataset selection was added.
run_imdb_example = run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare heterogeneous graph models on IMDB, DBLP, or ACM')
    parser.add_argument('--dataset', type=str.lower, choices=['imdb', 'dblp', 'acm'], default='imdb')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Dataset cache directory (default: data/hgb/<dataset>)')
    parser.add_argument('--model', choices=['aggregator-ablation', 'subpath', 'no-context', 'metapath-bias', 'metapath-scalar-bias', 'sehgnn', 'simplehgn', 'hettree', 'both', 'ablation', 'all'], default='both',
                        help='aggregator-ablation: all four subpath encoders; both: subpath and SeHGNN; ablation: subpath, no context, vector bias, and scalar bias; all: all seven models')
    parser.add_argument('--train-ratio', type=parse_training_budget, default=0.01,
                        help='Training share of target nodes, e.g. 0.20 or 20%%; use full for every '
                             'training-pool label while keeping validation/test fixed')
    parser.add_argument('--split-protocol', choices=['fixed', 'legacy'], default='fixed',
                        help='fixed: same validation/test nodes and nested training sets across ratios; '
                             'legacy: reproduce the previous ratio-dependent splits')
    test_size = parser.add_mutually_exclusive_group()
    test_size.add_argument('--test-ratio', type=parse_train_ratio, default=0.10,
                           help='Fixed test share of target nodes (default: 10%%); used by the fixed protocol')
    test_size.add_argument('--test-count', type=int, default=None,
                           help='Exact number of test target nodes, e.g. 800; replaces the default test ratio; fixed protocol only')
    parser.add_argument('--seed', type=int, default=42, help='Split and model initialization seed')
    parser.add_argument('--epochs', type=int, default=249)
    parser.add_argument('--subpath-variant', choices=['original', 'regularized'], default='original')
    parser.add_argument('--subpath-encoder', choices=['concat-mlp', 'mean', 'lstm', 'transformer'],
                        default='concat-mlp', help='Encoder for the ordered subpath type embeddings')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=None, help='Dropout for the subpath and bias encoders')
    parser.add_argument('--hettree-hops', type=int, default=None,
                        help='Depth limit covering the shared metapaths (default: 4 for IMDB/DBLP, 2 for ACM); larger limits add no paths')
    parser.add_argument('--hettree-no-labels', action='store_true',
                        help='Run HetTree using features only, with no training-label inputs')
    parser.add_argument('--selection-metric', choices=['loss', 'f1_macro'], default='loss',
                        help='Validation metric used to select the checkpoint; test metrics are reported only at the end')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=Path, default=None)
    parser.add_argument('--ablation-seeds', nargs='+', type=int, default=None,
                        help='Seeds for aggregator-ablation (defaults to --seed)')
    args = parser.parse_args()
    if args.ablation_seeds is not None and args.model != 'aggregator-ablation':
        parser.error('--ablation-seeds requires --model aggregator-ablation')
    runner = run_aggregator_ablation if args.model == 'aggregator-ablation' else run_experiment
    extra = {'seeds': args.ablation_seeds} if args.model == 'aggregator-ablation' else {}
    runner(
        **extra,
        dataset_name=args.dataset, data_root=args.data_root,
        train_ratio=args.train_ratio, split_seed=args.seed, model_name=args.model,
        epochs=args.epochs, patience=args.patience, checkpoint_dir=args.checkpoint_dir,
        split_protocol=args.split_protocol, test_ratio=args.test_ratio, test_count=args.test_count,
        subpath_encoder=args.subpath_encoder,
        subpath_variant=args.subpath_variant, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, dropout=args.dropout, selection_metric=args.selection_metric,
        hettree_hops=args.hettree_hops, hettree_use_labels=not args.hettree_no_labels,
    )
