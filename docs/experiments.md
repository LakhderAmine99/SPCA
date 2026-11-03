# Additional experiment commands

This guide covers the `index.py` interface, model options and ablation commands.
For the dataset runners and grid search, see [experiment settings](paper_protocol.md)
and the [README](../README.md).

## Dataset selection

Select the dataset with `--dataset imdb`, `--dataset dblp`, or `--dataset acm`.
The default is `imdb`. Every model option, percentage training budget, and the
`full` budget work with each dataset:

```powershell
python index.py --dataset imdb --model sehgnn --train-ratio "10%"
python index.py --dataset dblp --model sehgnn --train-ratio "20%"
python index.py --dataset acm --model subpath --train-ratio "1%"
python index.py --dataset dblp --model all --train-ratio full
```

| Dataset | Prediction target | Classes | Shared metapaths for all models |
| --- | --- | ---: | --- |
| IMDB | Movie genre | 3 | Movie–actor–movie, movie–director–movie, and their four-hop combinations |
| DBLP | Author research area | 4 | Author–paper–author, author–paper–term–paper–author, author–paper–conference–paper–author |
| ACM | Paper research area | 3 | Paper–author–paper, paper–subject–paper |

Percentages refer to the selected dataset's target nodes. The default fixed
protocol reserves the dataset's validation count and 10% of its target nodes
for test, then draws nested training subsets from the remaining pool. All models
in a comparison share the same masks and use `metapath_configuration(dataset_name)`:
**MAM, MDM, MAMDM, MDMAM** for IMDB; **APA, APTPA, APVPA** for DBLP;
and **PAP, PSP** for ACM. Here V is venue (named `conference` by the loader)
and S is subject. SeHGNN retains raw target features and computes only these
requested propagated channels, with a four-hop limit for IMDB/DBLP and two for ACM.
Simple-HGN is a metapath-based adaptation on every dataset: it receives a
target-only graph with one edge type per shared metapath. An edge means that
a path instance connects its two target nodes; multiple instances of the same
metapath form one binary edge.
Simple-HGN retains its typed self-loops and attention layers, so successive
layers can combine the relation types. Reports record the input as
`metapath-projected`, and run names include `metagraph`.
Dense metapath graphs use an equivalent matrix aggregation, with recomputation
during backward to limit storage; sparse graphs use chunked messages.
HetTree uses the same propagated feature paths and the raw target root. Required
ancestor nodes carry zero input features, so they connect the tree without
introducing extra propagated feature channels. Its optional training-label
propagation is limited to the shared metapaths.
Reports record `shared_metapaths` and each model's selected paths.
Checkpoint/report names include `sharedpaths_v2`. Each model uses its own
architecture to compose information across layers.

Data downloads and processed graphs are cached in `data/hgb/<dataset>/`.
Use `--data-root "C:/datasets/dblp"` to choose the cache directory for a dataset.
An existing `data/dblp/` cache is reused when the default DBLP cache does not exist.
Checkpoint and report names start with the selected dataset, preventing an ACM
or DBLP run from overwriting an IMDB result. The Python API is
`run_experiment(dataset_name='dblp', ...)`; `run_imdb_example(...)` remains
available for existing notebooks. Use `run_imdb.py --phase tune` to tune IMDB.

DBLP supports the
[PyG/MAGNN data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html).
Place its seven raw files or `DBLP_processed.zip` in `<data-root>/raw/` to use
prepared features. The loader also supports graph construction from pinned
MAGNN raw sources, with paper-title bag-of-words features of up to 2,048 words,
neighbor means for authors/terms, and identity features for conferences.
These runs use `raw_title` in their filenames. Reports record the feature source,
preprocessing settings and node counts.

ACM uses the 4,025-paper raw variant from
[DGL's HAN preprocessing](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py),
with paper words and mean paper features for authors/subjects. The conference
matrix supplies paper labels and is excluded from graph edges and features.
This loader uses PyG/SciPy directly and does not require DGL. The ACM validation
count is 402; IMDB and DBLP each use 400. All datasets download automatically on
first use and can run from their caches afterward.

## IMDB comparison

Run the subpath model and SeHGNN on the same 1% training split:

```powershell
python index.py
```

Use `--model sehgnn`, `--model subpath`, `--model no-context`, `--model metapath-bias`,
`--model metapath-scalar-bias`, `--model simplehgn`, or `--model hettree` to run one model.
Use `--model ablation` to compare subpath encoding with no context and with
vector and scalar metapath biases. Use `--model all` to compare all seven;
`--model both` keeps the subpath/SeHGNN comparison.
For example:

```powershell
python index.py --model sehgnn --train-ratio 0.01 --seed 42
python index.py --model simplehgn --train-ratio 0.01 --seed 42
python index.py --model hettree --train-ratio "10%" --seed 42
python index.py --model all --train-ratio 0.01
python index.py --model sehgnn --train-ratio "10%"
python index.py --model sehgnn --train-ratio "20%"
python index.py --model sehgnn --train-ratio "1%"
python index.py --model both --train-ratio 0.05
python index.py --model both --epochs 2
python index.py --model ablation --train-ratio "1%" --seed 42
python index.py --model no-context --train-ratio "1%" --seed 42
python index.py --model metapath-bias --train-ratio "1%" --seed 42
python index.py --model metapath-scalar-bias --train-ratio "1%" --seed 42
```

All models use the PyG/MAGNN three-class IMDB dataset, the same training,
validation, and test masks, Adam (learning rate 0.005, weight decay 0.0005), and
early stopping by validation loss. Ratios are fractions of all 4,278 movies and
accept either decimals (`0.10`) or percentages (`"10%"`).
The default `--split-protocol fixed` first reserves stratified validation and
test sets independently of the requested training ratio. For a given seed it
always uses the same 400 validation movies and 428 test movies (10% of all
movies). The remaining 3,450 movies form the training pool. A reproducible
ordering of this pool maintains approximately its class proportions, and each
training budget takes a prefix of that ordering. Thus 20% training is a subset
of 60% training, which is a subset of 80% training. Unselected training-pool
labels remain unused rather than moving into the test set.

| Training ratio | Training movies | Validation movies | Test movies | Unused labels |
| --- | ---: | ---: | ---: | ---: |
| 1% | 43 | 400 | 428 | 3,407 |
| 20% | 856 | 400 | 428 | 2,594 |
| 60% | 2,567 | 400 | 428 | 883 |
| 80% | 3,422 | 400 | 428 | 28 |
| full | 3,450 | 400 | 428 | 0 |

To use every label outside the fixed validation/test sets, select
`--train-ratio full`. This trains on 3,450 movies (about 80.65% of all movies)
and leaves no unused labels, while keeping exactly the same validation/test
movies as the percentage-budget runs:

```powershell
python index.py --model subpath --train-ratio full --seed 42
```

The `full` budget is available with the fixed protocol and uses `full` in
checkpoint/result filenames. JSON reports record both the requested budget and
the effective training fraction. Numeric percentages keep their existing
meaning: `80%` selects 3,422 movies, whereas `full` selects all 3,450 eligible
movies. Lower percentage budgets intentionally leave some training-pool labels
unused so they can be compared on the same held-out movies.

Use the same `--seed` and `--test-ratio` when comparing training ratios. The
default pool supports up to 3,450 training movies (about 80.65% of all movies).
Use `--test-ratio "5%"` if a larger training budget is needed; this changes the
held-out test set, so rerun every comparison ratio with that setting. The graph
and node features remain available during message passing; only the labels
selected by the training mask contribute to the training loss.

Use `--test-count 800` to reserve exactly 800 target nodes for testing on any
dataset and model, including HetTree without label inputs:

```powershell
python index.py --dataset dblp --model hettree --train-ratio "60%" --test-count 800 --hettree-no-labels
```

Choose either `--test-count` or `--test-ratio`. If neither is supplied, the
default test share remains 10%. Exact counts require the `fixed` split protocol.
Keep the test count and seed identical when comparing models or training budgets;
validation/test membership stays fixed and training subsets remain nested.
Validation counts remain unchanged. The chosen training budget must fit outside
validation and test; `--train-ratio full` uses all remaining training-pool labels.
For DBLP with 60% training and 800 test nodes, the counts are 2,434 training,
400 validation, 800 test, and 423 unused labels.
The Python API accepts `run_experiment(..., test_count=800)`, where an explicit
count overrides `test_ratio`.

For example, run a controlled label-budget comparison with:

```powershell
foreach ($ratio in "20%", "60%", "80%") {
    python index.py --model subpath --train-ratio $ratio --seed 42
}
```

Use fixed holdouts and nested training sets for label-budget comparisons.
Repeat the curve across multiple seeds and summarize the variation.

Use `--split-protocol legacy` to reproduce the earlier behavior: small ratios
sample the original 400-movie training pool, while larger ratios rebuild all
three splits and change validation/test membership with the training ratio.

SeHGNN is a feature-only PyTorch adaptation of the
[authors' implementation](https://github.com/ICT-GIMLab/SeHGNN/tree/e92bd37d0b803457339555684f139b4c8f3e160d).
It precomputes mean aggregation for raw target features and the shared metapaths,
then learns per-metapath projections and transformer semantic fusion. The local
configuration uses 128 hidden channels, two feature projection layers, four task
layers, and dropout 0.5. Label propagation and multi-stage training are disabled.
The subpath encoder now has an explicit three-class prediction head.

Simple-HGN is adapted from the
[HGB node classification implementation](https://github.com/THUDM/HGB/tree/ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe/NC/benchmark/methods/baseline).
In this comparison it uses the projected metapath graph, one self-loop per node, learned edge-type
embeddings, joint attention over all incoming relations, node residuals,
residual attention between hidden layers, and L2-normalized class scores.
There are two hidden attention layers with eight heads and 64 channels per head,
a single-head output layer, 64-dimensional edge embeddings, feature/attention
dropout 0.5, LeakyReLU slope 0.05, and attention residual weight 0.05.
Its PyTorch/PyG operators run without DGL. The dataset and optimizer settings are
those of this shared experiment.

## HetTree comparison

HetTree implements equations (1)--(6) in
[Heterogeneous Graph Neural Network on Semantic Tree, AAAI 2025](https://doi.org/10.1609/aaai.v39i16.33860).
The configuration uses two-layer MLPs, 128 hidden channels, dropout 0.5,
PReLU in MLPs, LeakyReLU attention scores, and ELU for child aggregation.
It runs in the PyTorch/PyG environment.

The model precomputes relation-wise mean feature aggregation, matches features
and propagated training labels on the same metapath, builds the semantic tree,
and applies bottom-up subtree attention. Its readout sums all encoded tree
nodes and adds raw-feature and propagated-label residual classifiers.

HetTree uses `metapath_configuration(dataset_name)`, the shared definitions used
by the subpath models: MAM, MDM, MAMDM, and MDMAM for IMDB; APA, APTPA, and
APVPA for DBLP; PAP and PSP for ACM (S is subject). The tree contains these
selected paths, their necessary intermediate ancestors, and the raw target root.
Only selected paths and the root carry input features; other ancestors have
zero inputs and serve as tree structure. It does not enumerate other paths
within the depth limit. In source-to-target
notation, each tree node's parent is its suffix after removing the source type.
This produces nine tree nodes for IMDB, nine for DBLP, and five for ACM.

The resulting depth is four hops for IMDB/DBLP and two for ACM. Reports record
the selected metapaths and active feature paths separately from the full tree
and zero-input ancestor paths, along with label paths, tree size, depth, and
implementation version.

Only labels selected by the current training mask enter propagation; validation,
test, and unused labels are zeroed. For each closed metapath, the exact diagonal
of its propagation operator is removed, then the remaining mass is normalized,
so a node cannot receive its own label through return walks. The diagonal is
computed in bounded-width blocks without storing a dense target-by-target
matrix. No pseudo-labels or multi-stage training are used.

```powershell
python index.py --dataset imdb --model hettree --train-ratio "10%"
python index.py --dataset dblp --model hettree --train-ratio "10%"
python index.py --dataset acm --model hettree --train-ratio "5%"
```

Use `--hettree-no-labels` for a feature-only comparison with the existing
feature-only SeHGNN baseline. It removes label concatenation and the label
residual. `--hettree-hops` is an optional depth limit that must accommodate all
shared metapaths: at least four for IMDB/DBLP and two for ACM. Larger limits do
not introduce additional paths.

```powershell
python index.py --dataset imdb --model hettree --train-ratio "10%" --hettree-no-labels
```

`--model all` includes HetTree; `both` and `ablation` retain their existing
members. All models use the same masks, training loop, optimizer settings, and
validation checkpoint selection. HetTree checkpoint/report names include
`sharedpaths_tree<depth>h_labels` or `sharedpaths_tree<depth>h_features` to keep
these settings separate from previous runs using all metapaths.

## Checkpoints and reports

Final metrics come from each model's best validation checkpoint. Checkpoints are
saved separately by model, split protocol, training ratio, and seed under `data/checkpoints/`.
Each run also writes its configuration and metrics to a JSON results file there.
Fixed-protocol filenames include `fixed_test10pct` (or the requested test share),
or `fixed_test800nodes` when using an exact test count,
to distinguish them from earlier runs. JSON reports include the exact train,
validation, and test node indices as well as unused-label and training-pool counts.
The Python function `run_imdb_example(...)` also returns the comparison metrics
for use in a notebook. Change `--epochs`, `--patience`, or `--checkpoint-dir` as needed.

## Subpath encoding ablation

`no-context` removes the context branch entirely: it has no type embeddings,
subpath MLPs, or metapath bias parameters, and adds no context to the metapath
embeddings. It keeps message passing, normalization, fusion, the classifier,
and their ordinary layer biases. Run it alone on DBLP with:

```powershell
python index.py --dataset dblp --model no-context --train-ratio "10%"
```

It also supports `--subpath-variant regularized`. Both `--model ablation` and
`--model all` include this variant. Its checkpoints use the `no-context` model
name, and its report records zero context parameters and no context injection.

The bias variants replace the subpath encoder's context with one independent,
trainable parameter per metapath, shared across all its nodes. `metapath-bias`
uses a vector; `metapath-scalar-bias` uses a single scalar broadcast across all
embedding dimensions:

```text
subpath:              z_m(v) = h_m(v) + mean_s SubpathEncoder(s)
no-context:           z_m(v) = h_m(v)
metapath-bias:        z_m(v) = h_m(v) + b_m       (b_m has 128 entries)
metapath-scalar-bias: z_m(v) = h_m(v) + s_m * 1   (s_m is a learned scalar)
```

Both kinds of bias are `nn.Parameter` values initialized to zero and learned by
Adam through the classification loss. Neither bias variant has type embeddings
or subpath MLPs. Across all four ablations, message passing, normalization,
concatenation and linear fusion, and the classification head are identical.
Runs use the same data masks and exactly the same initial weights for all shared
layers, including when a variant runs alone with the same seed. All four use the existing optimizer and
validation-based checkpoint selection. The console reports total trainable
parameters; JSON results also record context parameter counts and configuration.
With `M` metapaths, vector bias has `128 * M` context parameters and scalar bias
has `M`. The current four-metapath setting therefore has 512 vector-bias context
parameters or 4 scalar-bias context parameters. Scalar and vector checkpoints
use separate model names and filenames.

Run several seeds to assess variation:

```powershell
foreach ($seed in 42, 43, 44, 45, 46) {
    python index.py --model ablation --train-ratio "1%" --seed $seed
}
```

Each seed controls both the split and initialization. Compare paired test
F1-macro scores and summarize them across seeds.

For direct encoder use, pass `context_mode='no-context'`, `context_mode='metapath-bias'`, or
`context_mode='metapath-scalar-bias'` to `SubpathHANConv2` or `StackedSubpathHAN`.
`RegularizedSubpathHANConv` supports the same modes. No-context and both bias
modes support `subpaths_dict=None`.
The default remains `context_mode='subpath'`.



### Alternative subpath encoders

Select `--subpath-encoder concat-mlp` (default), `mean`, `lstm`, or `transformer`.
This option works with both `--subpath-variant original` and `regularized`.

- `lstm`: a separate single-layer LSTM for each distinct subpath reads its ordered
  sequence of learned type embeddings; its final hidden state represents the subpath.
- `transformer`: a separate Transformer encoder layer and learned positional
  embeddings for each distinct subpath, with mean pooling over output positions.
  It uses four attention
  heads for the default 128 channels (two or one when required by channel size),
  a feedforward width of twice the embedding size, and the configured dropout.

Both alternatives average the encoded subpaths for each metapath, as before.
Like concat-MLP, each distinct type sequence has its own encoder; if that same
subpath appears in multiple metapaths, it reuses its encoder. Type embeddings
remain shared across subpaths. Context is shared across nodes. No-context and
bias ablations bypass the subpath encoder.
Encoder choices are recorded in reports and non-default choices have distinct
checkpoint/report filenames. These options are exposed by `SubpathHANConv2`,
`RegularizedSubpathHANConv`, and `StackedSubpathHAN` via `aggregator=`.
Sequence encoders and ablation runs use `persubpath_v2` checkpoint filenames;
ablation report configuration includes `encoder_protocol=per-subpath-v2`.

```powershell
python index.py --model subpath --subpath-encoder lstm
python index.py --model subpath --subpath-encoder transformer --subpath-variant regularized
```

### Aggregator ablation

Run all four aggregators sequentially with identical splits, type embeddings,
shared message-passing/fusion/classifier weights, and training settings:

```powershell
python index.py --model aggregator-ablation --dataset imdb --train-ratio "1%"
python index.py --model aggregator-ablation --dataset imdb --train-ratio "10%" --ablation-seeds 42 43 44 45 46
```

Use `--subpath-variant regularized` to compare them with the regularized backbone.
This mode always runs concat-MLP, mean, LSTM, and the one-layer Transformer;
`--subpath-encoder` is ignored. Each seed resets the split and initialization.
Validation selects each model's checkpoint; test labels are evaluated only after
training. Subpath encoder parameters differ, while common initial weights match.
Reports record the encoder sharing policy and number of distinct encoders.

Reports are saved in `--checkpoint-dir` (default `data/checkpoints`):

- `*_aggregator_ablation_*.csv`: one row per aggregator/seed, including test
  accuracy, micro/macro F1, test loss, total/context parameter counts, best epoch,
  best validation loss, epochs run, device, checkpoint, and timing statistics.
- `*_aggregator_ablation_*_summary.csv`: means and sample standard deviations
  across seeds (blank standard deviation for a single seed).
- `*_aggregator_ablation_*.json`: full results, individual epoch timings,
  configuration, summaries, timing definitions, and total sweep wall time.
  Completed runs are saved progressively with `complete=false` until finished.

Scores in saved files are fractions; the console summary shows percentages.
Epoch timing includes training and train/validation evaluation, with CUDA
synchronization when applicable. It excludes preprocessing, checkpoint I/O,
logging, and final testing. All executed epochs, including the first, contribute
to the mean, population standard deviation, median, minimum and maximum. Training
step mean time and summed epoch time are also reported. Across-seed summaries
average each run's epoch mean equally. Different early-stopping lengths can
therefore affect total time. No warm-up epochs are discarded.
