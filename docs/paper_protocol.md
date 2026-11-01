# Model and experiment settings

This document describes the SPCA method, dataset preparation, hyperparameter search and evaluation settings. Use `run_<dataset>.py` to configure experiments. Additional commands for `index.py` are documented in [experiments.md](experiments.md).

## Method

| Manuscript | Code | Behavior |
| --- | --- | --- |
| Section 4.1, Eqs. 1–2 | `spca/layers.py:DotProductConv` | Full-vector, unscaled dot product; LeakyReLU; destination softmax; coefficient dropout; weighted endpoint sum |
| Section 4.1 | `SPCALayer.structural_norm` | Normalize the structural aggregate |
| Section 4.2, Eqs. 3–4 | `SubpathContext` | Shared type embeddings; one encoder per distinct ordered sequence; intermediate positions by default |
| Section 4.3, Eq. 5 | `SPCALayer.fusion_norm` | Add schema context, then normalize each receiving node |
| Section 4.4, Eqs. 6–7 | `SPCALayer.fusion` | Concatenate selected metapath outputs and linearly project |
| Section 4.5, Eq. 8 | `experiments/models.py:NodeTask` | Linear classifier and training-mask cross entropy |
| Section 4.5, Eq. 10 | `LinkTask` | Separate endpoint encoders and a dot-product logit; BCE with logits |

Structural attention uses unique endpoint neighbors; intermediate node features do not undergo hop-by-hop updates in SPCA. A reachable self-neighbor is retained, but no additional self-loop is inserted by SPCA. Other models retain their native self-loop conventions. Binary adjacency composition uses Boolean sparse matrices to avoid path-count overflow. SPCA's attention can process destination blocks without changing neighborhood softmax or the mathematical result; recomputation during backward bounds message memory.

Full-sequence context is available through `--set spca.context_positions=full`. `SPCALayer(subpaths_dict=...)` also accepts explicitly selected ordered type sequences. Mean, LSTM and one-layer Transformer encoders, no-context, vector/scalar lookup biases, and removal of normalization are explicit options. Additional hop-wise encoders are available in `spca/encoders.py`.

## Search and selection

Table 11 is encoded in `configs/grid.json`: learning rate `{0.0005, 0.001, 0.003, 0.005}`, weight decay `{0.0001, 0.0005}`, dropout `{0.1, 0.3, 0.5, 0.6}`, hidden dimension `{64, 128, 256, 512}`, and channels per attention head `{8, 16, 32, 64, 128}`. There are 128 base combinations and 640 combinations when a separate head-width axis applies. GAT, HAN, MAGNN, Simple-HGN and BG-HGNN expose that separate width. HGT, GTC and SeHGNN divide the hidden dimension across their heads. SPCA's Eq. 1 uses a single full-vector score, so its attention has no separate head-width axis.

The default attention head count is eight, and the final embedding width is 128 by default. A supplied `--set` fixes that parameter and removes it from the search. All remaining applicable axes are enumerated exhaustively. Model-specific settings are centralized in `model_defaults()` and recorded per trial.

Three tuning seeds are used for every configuration. Within each seed, training uses Adam and validation runs after every epoch, up to 500 epochs, stopping after 100 consecutive epochs without improvement. Higher validation Macro-F1 wins; lower validation loss breaks exact ties. Configurations maximize mean selected-checkpoint validation Macro-F1. Mean validation loss breaks configuration ties; first grid order breaks remaining exact ties. Five reporting seeds then train fresh models with the selected configuration. Test evaluation is separate from `fit()` and runs once at the selected checkpoint. Reports use sample standard deviation (`ddof=1`); one-run reports use `null` for the undefined sample deviation.

Default seeds are `[0, 1, 2]` for tuning and `[42, 43, 44, 45, 46]` for reporting; both are configurable. `--phase report` uses saved selections from `--selected-config` when supplied, or the configured fixed settings otherwise. `--phase tune` evaluates validation metrics.

## Splits and data

Main NC ratios (20%, 40%, 60%) use the **existing training pool within each class**. Low-label ratios (5%, 10%) use **all labeled nodes per class**, sampled outside fixed validation/test sets. Both select deterministic per-class prefixes, so larger budgets contain smaller budgets for the same seed. Counts use nearest-integer rounding with at least one training example per class; impossible budgets fail. Loader holdouts remain fixed across models and reporting seeds. Explicit holdout overrides use a fixed `--split-seed` independently of model/training-label seeds. Label-derived HetTree features use only the selected training labels and remove self-label return paths.

LastFM uses a 70/10/20 interaction split; MovieLens-1M uses 80/10/10. All observed target positives are collected before splitting. Held-out target interactions are removed in both directions before any encoder preparation. Negatives are sampled without replacement from pairs absent from the complete observed graph, one per positive, and are disjoint across the three splits. LP graph caches are built anew per split. LP checkpoints maximize validation ROC-AUC, with lower BCE breaking ties, using the same patience and epoch limit. AP is also reported.

PyG/MAGNN DBLP and IMDB caches are supported. Use `--graph` to supply a prepared local PyG graph. NC graphs contain typed `x` features and `edge_index` tensors, target `y` labels, and boolean `train_mask`, `val_mask` and `test_mask` tensors. Files passed to `--graph` must be trusted because PyG graph loading uses Python object deserialization.

LastFM uses the MAGNN/PyG processed HetRec graph (1,892 users, 17,632 artists, 1,088 tags). PyG's supplied held-out positives are reunited with observed interactions before applying this protocol. Fixed Gaussian features use a configurable width and feature seed, shared across models. MovieLens-1M retains all 3,883 metadata movies, 6,040 users and 18 genres. Inputs use user demographic categories, movie genre indicators and genre identities. Feature settings are recorded in the manifest.

K-means uses the selected NC embeddings, the known number of classes and the reporting seed, with `n_init=10`. It evaluates all target nodes by default; `--cluster-split test` restricts evaluation to held-out nodes.

## Baseline configuration

Model folders contain the architectures, reusable layers and model-specific preprocessing. Upstream snapshots provide original source and preprocessing scripts with pinned versions. The shared runner prepares inputs with PyG/SciPy so all models receive the same graph, masks and Table 3 paths.

- GCN/GAT use the union of induced endpoint graphs. RGCN, HGT, Simple-HGN and BG-HGNN retain one relation ID per selected metapath. Type-aware models operate on one receiving node type per endpoint encoder.
- MAGNN uses ordered instance features and RotatE0 by default. It samples up to 100 complete instances per receiver using prefix-count sampling with replacement when needed. All instances are available with `--set magnn.max_instances=null`.
- HeCo uses metapath/schema views and a multi-positive contrastive loss. Its schema view receives the selected metapath relations, uses eight heads, and aggregates all supplied neighbors. Semantic positives come from top-ranked selected-path neighbors plus self pairs.
- GTC uses Hop2Token, a Transformer per metapath, semantic fusion, a relation-wise GNN branch and co-contrastive loss. `transformer.py` provides the Transformer layers with configurable head count and depth, supporting singleton batches.
- For HeCo/GTC NC, contrastive representation learning and a detached linear evaluation head advance together each epoch. The encoder receives the contrastive loss; output/evaluation projections receive cross entropy. For LP, BCE and the contrastive objective train jointly.
- SeHGNN uses selected mean-propagated feature channels and raw target features, per-path MLPs, gated Transformer semantic fusion and residual task layers.
- HetTree completes selected paths with zero-input ancestor nodes to preserve tree connectivity. It combines training-label matching, subtree attention and feature/label residuals. Terminal output projections use the requested embedding width.
- BG-HGNN uses random type/relation encodings and low-rank multiplicative fusion before a shared homogeneous GAT. Feature construction uses the configured random seed and device.

Original preprocessing scripts are included in the baseline source snapshots.

## Local output and verification

Every executed run writes a manifest with software versions, effective options, selected paths, node/edge counts, and a SHA-256 graph/features/labels/splits fingerprint. Tuning records retain validation metrics; reporting records add the test evaluation. Checkpoint saving is opt-in. A new timestamped `local/experiments/` directory is used by default.

Unit tests cover SPCA's manual equation values, internal-only context, endpoint feature aggregation, shared sequence encoders, scalar-shift cancellation, graph composition, all model/dataset adapters, synthetic optimizer steps, LP splits/decoding, selection ties, sample standard deviation and CLI dry runs.
