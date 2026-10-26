"""Select a subpath configuration on validation data, then evaluate test once."""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from index import (
    RegularizedSubpathHANConv, SubpathClassifier, SubpathHANConv2,
    imdb_metapath_configuration, make_imdb_masks, parse_train_ratio, test, train,
)
from src.checkpoint import save_checkpoint
from src.imdb import IMDB


CANDIDATES = [
    {'variant': 'original', 'dropout': 0.0, 'learning_rate': 0.005, 'weight_decay': 5e-4},
    {'variant': 'original', 'dropout': 0.3, 'learning_rate': 0.001, 'weight_decay': 5e-4},
    {'variant': 'regularized', 'dropout': 0.3, 'learning_rate': 0.001, 'weight_decay': 5e-4},
    {'variant': 'regularized', 'dropout': 0.5, 'learning_rate': 0.001, 'weight_decay': 5e-4},
    {'variant': 'regularized', 'dropout': 0.5, 'learning_rate': 0.003, 'weight_decay': 5e-4},
]


def run_validation_search(train_ratio=0.60, seed=42, epochs=200, patience=40, output_dir=None):
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    root = Path(__file__).resolve().parent
    output_dir = Path(output_dir).resolve() if output_dir else root / 'data' / 'tune60'
    output_dir.mkdir(parents=True, exist_ok=True)
    data = IMDB(root=str(root / 'data' / 'hgb' / 'imdb'))[0]
    masks, strategy = make_imdb_masks(
        data['movie'].y, data['movie'].train_mask, data['movie'].val_mask,
        data['movie'].test_mask, train_ratio, seed=seed,
    )
    for name, mask in masks.items():
        data['movie'][name] = mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    metapaths, subpaths = imdb_metapath_configuration()
    channels = {key: value.size(-1) for key, value in data.x_dict.items()}
    classes = int(data['movie'].y.max()) + 1
    tag = f'{train_ratio * 100:g}pct_seed{seed}'
    trials = []
    selected = None
    for trial_id, config in enumerate(CANDIDATES):
        torch.manual_seed(seed)
        encoder_class = RegularizedSubpathHANConv if config['variant'] == 'regularized' else SubpathHANConv2
        model = SubpathClassifier(encoder_class(
            channels, 128, data.metadata(), metapaths, subpaths,
            heads=8, dropout=config['dropout'],
        ), 128, classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
        best_key = (-float('inf'), -float('inf'))
        stale = 0
        best_state = None
        history = []
        for epoch in range(1, epochs + 1):
            train_loss = train(model, data, optimizer, nn.CrossEntropyLoss(), device)
            metrics = test(model, data, device, ('train_mask', 'val_mask'))
            validation = metrics['val_mask']
            history.append({'epoch': epoch, 'train_loss': train_loss, 'validation': validation})
            key = (validation['f1_macro'], -validation['loss'])
            if key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 25 == 0:
                print(f'Trial {trial_id} {config} | epoch {epoch} | '
                      f"validation macro {validation['f1_macro']:.4f} | best {best_key[0]:.4f}", flush=True)
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError(f'Trial {trial_id} did not produce a checkpoint')
        checkpoint = save_checkpoint(best_state, output_dir / f'{tag}_trial{trial_id}.pth')
        trial = {
            'trial_id': trial_id, 'configuration': config, 'best_epoch': best_epoch,
            'checkpoint': str(checkpoint), 'metrics': best_metrics, 'history': history,
        }
        trials.append(trial)
        print(f"Finished trial {trial_id}: validation macro {best_key[0]:.4f}", flush=True)
        if selected is None or best_key > selected[0]:
            selected = best_key, trial_id, best_state

    # The choice is final before test labels or test metrics are requested.
    _, trial_id, state = selected
    config = CANDIDATES[trial_id]
    encoder_class = RegularizedSubpathHANConv if config['variant'] == 'regularized' else SubpathHANConv2
    winner = SubpathClassifier(encoder_class(
        channels, 128, data.metadata(), metapaths, subpaths, heads=8,
        dropout=config['dropout'],
    ), 128, classes).to(device)
    winner.load_state_dict(state)
    final_metrics = test(winner, data, device)
    report = {
        'dataset': 'PyG/MAGNN IMDB (three-class)', 'train_ratio': train_ratio,
        'seed': seed, 'split_strategy': strategy,
        'split_indices': {name: mask.nonzero(as_tuple=True)[0].cpu().tolist() for name, mask in masks.items()},
        'selection_metric': 'validation f1_macro, validation loss breaks ties',
        'selected_trial': trial_id, 'selected_configuration': config,
        'test_evaluations': 1, 'metrics': final_metrics, 'trials': trials,
    }
    report_path = output_dir / f'{tag}_validation_search.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Selected configuration:', config, flush=True)
    print('Final held-out test:', final_metrics['test_mask'], flush=True)
    print('Report:', report_path, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-ratio', type=parse_train_ratio, default=0.60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    run_validation_search(args.train_ratio, args.seed, args.epochs, args.patience, args.output_dir)
