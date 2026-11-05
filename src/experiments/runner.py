"""Configurable manuscript grid search and five-run reporting; no import effects."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from .config import MODELS, GRID, PATH_CODES, LP_RATIOS, configurations, model_defaults, ratio

log = logging.getLogger('hgnn')
_FORMAT = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')


def setup_logging(quiet=False):
    """Console logging to stderr (unbuffered); a run.log file is attached once the output folder is known."""
    log.setLevel(logging.WARNING if quiet else logging.INFO)
    log.propagate = False
    if not any(getattr(h, '_hgnn_console', False) for h in log.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_FORMAT)
        handler._hgnn_console = True
        log.addHandler(handler)


def attach_log_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding='utf-8')
    handler.setFormatter(_FORMAT)
    log.addHandler(handler)
    return handler


def make_parser(dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='JSON defaults; command-line flags take precedence')
    parser.add_argument('--dataset', choices=PATH_CODES, default=dataset or 'dblp')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--task', choices=['nc', 'clustering', 'lp'])
    parser.add_argument('--protocol', choices=['main', 'low-label'], default='main')
    parser.add_argument('--train-ratios', '--train-ratio', nargs='+', type=ratio)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=42, help='First reporting seed when --seeds is omitted')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--tuning-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--split-seed', type=int, default=42, help='Fixed custom NC holdouts, when explicitly requested')
    parser.add_argument('--val-count', type=int, help='Override the original fixed validation set')
    test = parser.add_mutually_exclusive_group()
    test.add_argument('--test-count', type=int)
    test.add_argument('--test-ratio', type=ratio)
    parser.add_argument('--lp-ratios', nargs=3, type=float, metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--negatives-per-positive', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--phase', choices=['all', 'tune', 'report'], default='all')
    parser.add_argument('--selected-config', type=Path, help='Use selected.json from a previous tuning run')
    parser.add_argument('--grid', type=Path, help='JSON parameter arrays replacing the Table 11 grid')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='[MODEL.]KEY=JSON',
                        help='Fix any model hyperparameter, e.g. spca.aggregator=mean or learning_rate=0.001')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--graph', type=Path, help='Trusted local PyG HeteroData file with features and NC masks')
    parser.add_argument('--metapaths', type=Path, help='JSON target -> list of node-type sequences')
    parser.add_argument('--allow-data-mismatch', action='store_true')
    parser.add_argument('--feature-seed', type=int, default=0)
    parser.add_argument('--feature-channels', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-checkpoints', action='store_true')
    parser.add_argument('--auxiliary-weight', type=float, default=1.0)
    parser.add_argument('--cluster-split', choices=['all', 'test'], default='all')
    parser.add_argument('--kmeans-n-init', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true', help='Print complete plan; never import torch, load data or train')
    parser.add_argument('--log-every', type=int, default=10, help='Log training progress every N epochs (0 = only per-fit summaries)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress logs (warnings and errors only)')
    return parser


def parse_overrides(values):
    result = {'all': {}}
    for expression in values:
        if '=' not in expression:
            raise ValueError('--set requires [MODEL.]KEY=VALUE')
        key, value = expression.split('=', 1)
        model, key = key.split('.', 1) if '.' in key else ('all', key)
        if model != 'all' and model not in MODELS:
            raise ValueError(f'Unknown model in --set: {model}')
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        result.setdefault(model, {})[key] = value
    return result


def parse_args(argv=None, dataset=None):
    parser = make_parser(dataset)
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        values = json.loads(preliminary.config.read_text(encoding='utf-8'))
        allowed = {a.dest for a in parser._actions} - {'help', 'config'}
        if set(values) - allowed:
            parser.error(f'Unknown config fields: {sorted(set(values) - allowed)}')
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if args.dataset not in PATH_CODES or not args.models or any(name not in MODELS for name in args.models):
        parser.error('Unknown dataset or model in configuration')
    if args.phase not in {'all', 'tune', 'report'} or args.protocol not in {'main', 'low-label'}:
        parser.error('Unknown phase or protocol in configuration')
    args.models = [m for m in dict.fromkeys(args.models) if m != 'spca'] + (['spca'] if 'spca' in args.models else [])
    args.task = args.task or ('lp' if args.dataset in LP_RATIOS else 'nc')
    if args.task not in {'nc', 'clustering', 'lp'}:
        parser.error('Unknown task in configuration')
    if (args.task == 'lp') != (args.dataset in LP_RATIOS):
        parser.error('Use nc/clustering for DBLP, IMDB, ACM; lp for LastFM or MovieLens')
    args.train_ratios = [ratio(r) for r in (args.train_ratios or ([0.05, 0.1] if args.protocol == 'low-label' else [0.2, 0.4, 0.6]))]
    args.seeds = args.seeds or list(range(args.seed, args.seed + args.runs))
    args.lp_ratios = args.lp_ratios or LP_RATIOS.get(args.dataset)
    if args.test_ratio is not None and (args.test_ratio == 'full' or not 0 < float(args.test_ratio) < 1):
        parser.error('--test-ratio must be a fraction strictly between zero and one')
    if args.log_every < 0:
        parser.error('--log-every must be zero or positive')
    if args.negatives_per_positive < 1 or args.feature_channels < 1 or args.kmeans_n_init < 1:
        parser.error('Negative-sampling ratio, feature width and K-means restarts must be positive')
    if args.auxiliary_weight < 0:
        parser.error('auxiliary_weight must be nonnegative')
    if not args.seeds or not args.tuning_seeds or args.runs < 1 or min(args.epochs, args.patience, args.threads) < 1:
        parser.error('Seeds must be nonempty; runs, epochs, patience and threads must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.tuning_seeds)) != len(args.tuning_seeds):
        parser.error('Seed lists must not contain duplicates')
    if args.lp_ratios and (min(args.lp_ratios) <= 0 or abs(sum(args.lp_ratios) - 1) > 1e-8):
        parser.error('LP ratios must be positive and sum to 1')
    args.overrides = parse_overrides(args.overrides)
    for name in args.models:
        values = {**args.overrides['all'], **args.overrides.get(name, {})}
        unknown = set(values) - model_defaults(name, args.dataset).keys()
        if unknown:
            parser.error(f'Unknown or inapplicable parameters for {name}: {sorted(unknown)}')
    for field in ('grid', 'selected_config', 'data_root', 'graph', 'metapaths', 'output'):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, Path(value))
    return args


def plan(args):
    grid = json.loads(args.grid.read_text()) if args.grid else GRID
    if set(grid) - set(GRID):
        raise ValueError(f'Unknown grid keys: {set(grid) - set(GRID)}; use --set for model-specific settings')
    jobs = {}
    for name in args.models:
        overrides = {**args.overrides['all'], **args.overrides.get(name, {})}
        count = sum(1 for _ in configurations(name, overrides, grid))
        jobs[name] = {'defaults': model_defaults(name, args.dataset), 'fixed_overrides': overrides,
                      'grid_configurations': count, 'tuning_fits_per_budget': count * len(args.tuning_seeds),
                      'reporting_fits_per_budget': len(args.seeds)}
    return {'dataset': args.dataset, 'task': args.task, 'protocol': args.protocol,
            'train_ratios': args.train_ratios if args.task != 'lp' else None,
            'lp_ratios': args.lp_ratios, 'tuning_seeds': args.tuning_seeds, 'reporting_seeds': args.seeds,
            'epochs': args.epochs, 'patience': args.patience, 'phase': args.phase,
            'grid': grid, 'models': jobs, 'metapaths': json.loads(args.metapaths.read_text()) if args.metapaths else PATH_CODES[args.dataset],
            'checkpoint_selection': 'validation Macro-F1, then lower loss' if args.task != 'lp' else 'validation ROC-AUC, then lower BCE',
            'test_policy': 'once per reporting run; never during tuning',
            'options': {k: v for k, v in vars(args).items() if k not in {'overrides'}}}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')


def run(args):
    manifest = plan(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, default=str))
        return manifest
    setup_logging(args.quiet)
    log.info('Dataset %s | task %s | protocol %s | budgets %s | models %s | phase %s',
             args.dataset, args.task, args.protocol, 'lp' if args.task == 'lp' else args.train_ratios,
             ', '.join(args.models), args.phase)
    for name in args.models:
        job = manifest['models'][name]
        log.info('Plan %s: %d grid configs x %d tuning seeds + %d reporting seeds per budget',
                 name, job['grid_configurations'], len(args.tuning_seeds), job['reporting_fits_per_budget'])
    log.info('Importing torch / torch_geometric ...')
    # Heavy dependencies and all data access occur only after the dry-run exit.
    import torch
    import numpy as np
    import torch_geometric
    from .data import load_dataset
    from .models import PreparedGraph, BoundEncoder, NodeTask, LinkTask
    from .splits import node_masks, link_split
    from .training import seed_everything, fit, evaluate_test, summarize
    from src.splits import make_fixed_masks
    from src.checkpoint import save_checkpoint
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        log.warning('CUDA requested but unavailable (torch %s); falling back to CPU', torch.__version__)
        device = torch.device('cpu')
    log.info('Device: %s | torch %s | PyG %s', device, torch.__version__, torch_geometric.__version__)
    root = args.data_root or Path('data') / 'paper' / args.dataset
    log.info('Loading %s from %s ...', args.dataset, root)
    loading = time.perf_counter()
    data = load_dataset(args.dataset, root, graph_path=args.graph, feature_seed=args.feature_seed,
                        feature_channels=args.feature_channels, allow_data_mismatch=args.allow_data_mismatch)
    log.info('Loaded in %.1fs: nodes %s', time.perf_counter() - loading,
             {t: data[t].num_nodes for t in data.node_types})
    output = args.output or Path('local/experiments') / (args.dataset + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'{output} already contains a run; choose a new --output to preserve it')
    file_handler = attach_log_file(output / 'run.log')
    log.info('Output folder: %s (log also written to run.log)', output)
    manifest.update(versions={'python': sys.version, 'torch': torch.__version__, 'pyg': torch_geometric.__version__},
        data={'nodes': {t: data[t].num_nodes for t in data.node_types},
              'edges': {str(k): v.size(1) for k, v in data.edge_index_dict.items()},
              'feature_source': data.get('feature_source', 'loader-provided'),
              'discrepancies': data.paper_data_discrepancies}, device=str(device))
    # Fingerprint graph, features, labels and original masks; selected configs cannot cross datasets silently.
    digest = hashlib.sha256()
    for node in data.node_types:
        for key in ('x', 'y', 'train_mask', 'val_mask', 'test_mask'):
            if key in data[node]:
                digest.update(f'{node}:{key}'.encode())
                digest.update(data[node][key].contiguous().cpu().numpy().tobytes())
    for relation, edges in data.edge_index_dict.items():
        digest.update(str(relation).encode())
        digest.update(edges.contiguous().cpu().numpy().tobytes())
    manifest['data_fingerprint'] = digest.hexdigest()
    protocol = {k: getattr(args, k) for k in ('protocol', 'split_seed', 'val_count', 'test_count',
                'test_ratio', 'lp_ratios', 'negatives_per_positive')}
    protocol['metapaths'] = manifest['metapaths']
    protocol_fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest['protocol_fingerprint'] = protocol_fingerprint
    dump(output / 'manifest.json', manifest)
    custom_paths = json.loads(args.metapaths.read_text()) if args.metapaths else None
    chosen = {'dataset': args.dataset, 'task': args.task, 'protocol': args.protocol,
              'data_fingerprint': digest.hexdigest(), 'protocol_fingerprint': protocol_fingerprint, 'selected': {}}
    if args.selected_config:
        chosen = json.loads(args.selected_config.read_text())
        if (chosen['dataset'], chosen['task'], chosen['protocol'], chosen['data_fingerprint']) != (
                args.dataset, args.task, args.protocol, digest.hexdigest()):
            raise ValueError('Selected configuration belongs to a different dataset/task/protocol/fingerprint')
        if chosen.get('protocol_fingerprint') != protocol_fingerprint:
            raise ValueError('Selected configuration uses different holdouts, metapaths or LP sampling settings')
    target = next(iter(PATH_CODES[args.dataset]))
    nc_prepared = None
    if args.task != 'lp':
        original = {key: data[target][key] for key in ('train_mask', 'val_mask', 'test_mask')}
        if args.val_count is not None or args.test_count is not None or args.test_ratio is not None:
            original = make_fixed_masks(data[target].y, 'full',
                args.val_count if args.val_count is not None else int(original['val_mask'].sum()),
                args.split_seed, args.test_ratio or 0.1, args.test_count)
        data = data.to(device)
        log.info('Preparing metapath graph for target %s ...', target)
        nc_prepared = PreparedGraph(data, args.dataset, target,
                                    custom_paths[target] if custom_paths else None)
    budgets = [None] if args.task == 'lp' else args.train_ratios
    summary = {}
    for budget in budgets:
        budget_key = 'lp' if budget is None else str(budget)
        chosen['selected'].setdefault(budget_key, {})
        summary[budget_key] = {}
        def one_fit(name, params, seed, reporting=False, label=''):
            seed_everything(seed)
            started = time.perf_counter()
            settings = {**model_defaults(name, args.dataset), **params}
            if args.task == 'lp':
                endpoints = list(PATH_CODES[args.dataset])
                graph, splits = link_split(data, (endpoints[0], 'to', endpoints[1]), args.lp_ratios,
                                          seed, args.negatives_per_positive)
                graph = graph.to(device)
                encoders = [BoundEncoder(name, PreparedGraph(graph, args.dataset, node,
                    custom_paths[node] if custom_paths else None), settings, seed=seed) for node in endpoints]
                model = LinkTask(*encoders).to(device)
                splits = {split: {k: v.to(device) for k, v in values.items()} for split, values in splits.items()}
                train, val, test = splits['train'], splits['val'], splits['test']
                split_info = {k: hashlib.sha256(v['edge_label_index'].cpu().numpy().tobytes()).hexdigest() for k, v in splits.items()}
            else:
                masks = node_masks(data[target].y, original, budget, seed, args.protocol)
                labels = data[target].y
                encoder = BoundEncoder(name, nc_prepared, settings, labels=labels,
                    train_mask=masks['train_mask'], num_classes=int(labels.max()) + 1, seed=seed)
                model = NodeTask(encoder, settings['embedding_channels'], int(labels.max()) + 1).to(device)
                train, val, test = ({'labels': labels, 'mask': masks[k + '_mask']} for k in ('train', 'val', 'test'))
                split_info = {k: {'count': int(v.sum()), 'sha256': hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()} for k, v in masks.items()}
            result = fit(model, train, val, settings, task='lp' if args.task == 'lp' else 'nc',
                         epochs=args.epochs, patience=args.patience, auxiliary_weight=args.auxiliary_weight,
                         log_every=0 if args.quiet else args.log_every, label=label)
            result.update(seed=seed, settings=settings, splits=split_info)
            metric = 'auc' if args.task == 'lp' else 'macro_f1'
            log.info('%s done: best val %s %.4f at epoch %d (%d epochs, %.1fs)', label, metric,
                     result['validation'][metric], result.get('best_epoch', 0), result.get('epochs_completed', 0),
                     time.perf_counter() - started)
            if reporting:
                result['test'] = evaluate_test(model, test, task='lp' if args.task == 'lp' else 'nc')
                if args.task == 'clustering':
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
                    with torch.no_grad():
                        embeddings = model.encoder().cpu().numpy()
                    selected = test['mask'].cpu().numpy() if args.cluster_split == 'test' else np.ones(len(embeddings), dtype=bool)
                    predicted = KMeans(n_clusters=int(labels.max()) + 1, random_state=seed,
                                       n_init=args.kmeans_n_init).fit_predict(embeddings[selected])
                    truth = labels.cpu().numpy()[selected]
                    result['test'].update(nmi=float(normalized_mutual_info_score(truth, predicted)),
                                          ari=float(adjusted_rand_score(truth, predicted)))
                if args.save_checkpoints:
                    result['checkpoint'] = str(save_checkpoint(model.state_dict(), output / budget_key / name / f'seed_{seed}.pt'))
                log.info('%s TEST %s', label, '  '.join(f'{k} {v:.4f}' for k, v in result['test'].items()))
            return result
        for name in args.models:
            overrides = manifest['models'][name]['fixed_overrides']
            if args.phase in {'all', 'tune'} and not args.selected_config:
                best_key, best_params = None, None
                total = manifest['models'][name]['grid_configurations']
                log.info('=== %s | budget %s | tuning %d configurations ===', name, budget_key, total)
                for index, params in enumerate(configurations(name, overrides, manifest['grid'])):
                    log.info('[%s %s] config %d/%d: %s', name, budget_key, index + 1, total, params)
                    trials = [one_fit(name, params, seed, label=f'[{name} {budget_key} cfg {index + 1}/{total} seed {seed}]')
                              for seed in args.tuning_seeds]
                    metric = 'auc' if args.task == 'lp' else 'macro_f1'
                    score = (float(np.mean([r['validation'][metric] for r in trials])),
                             -float(np.mean([r['validation']['loss'] for r in trials])))
                    dump(output / budget_key / name / 'tuning' / f'config_{index:04}.json', {'parameters': params, 'runs': trials})
                    if best_key is None or score > best_key:
                        best_key, best_params = score, params
                    log.info('[%s %s] config %d/%d mean validation %.6f (best so far %.6f)',
                             name, budget_key, index + 1, total, score[0], best_key[0])
                log.info('[%s %s] selected: %s', name, budget_key, best_params)
                chosen['selected'][budget_key][name] = {**model_defaults(name, args.dataset), **best_params}
                dump(output / 'selected.json', chosen)
            if args.phase in {'all', 'report'}:
                if args.selected_config and name not in chosen['selected'].get(budget_key, {}):
                    raise ValueError(f'Selected configuration missing {budget_key}/{name}')
                settings = {**chosen['selected'][budget_key].get(name, model_defaults(name, args.dataset)), **overrides}
                reports = []
                log.info('=== %s | budget %s | reporting on %d seeds ===', name, budget_key, len(args.seeds))
                for seed in args.seeds:
                    result = one_fit(name, settings, seed, reporting=True, label=f'[{name} {budget_key} report seed {seed}]')
                    reports.append(result)
                    dump(output / budget_key / name / f'seed_{seed}.json', result)
                summary[budget_key][name] = summarize([r['test'] for r in reports])
                dump(output / 'summary.json', summary)
                log.info('[%s %s] SUMMARY %s', name, budget_key, '  '.join(
                    f"{k} {v['mean']:.4f}" + (f" +/- {v['std']:.4f}" if v['std'] is not None else '')
                    for k, v in summary[budget_key][name].items()))
    log.info('Finished. Local outputs: %s', output)
    log.removeHandler(file_handler)
    file_handler.close()
    print(f'Local outputs: {output}')
    return summary


def main(argv=None, dataset=None):
    return run(parse_args(argv, dataset))


if __name__ == '__main__':
    main()
