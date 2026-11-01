# Manuscript protocol and implementation notes

This document maps the implementation to the supplied November 2026 SPCA manuscript. It describes executable settings, not new experimental results. `run_<dataset>.py` is the current interface; `index.py` and [experiments.md](experiments.md) describe historical runs and remain available for compatibility.

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

Full-sequence context is available through `--set spca.context_positions=full`. `SPCALayer(subpaths_dict=...)` also accepts explicitly selected ordered type sequences. Mean, LSTM and one-layer Transformer encoders, no-context, vector/scalar lookup biases, and removal of normalization are explicit options. Historical hop-wise encoders are exported for compatibility; they are not the default SPCA implementation.

## Search and selection

Table 11 is encoded in `configs/grid.json`: learning rate `{0.0005, 0.001, 0.003, 0.005}`, weight decay `{0.0001, 0.0005}`, dropout `{0.1, 0.3, 0.5, 0.6}`, hidden dimension `{64, 128, 256, 512}`, and channels per attention head `{8, 16, 32, 64, 128}`. There are 128 base combinations and 640 combinations when a separate head-width axis applies. GAT, HAN, MAGNN, Simple-HGN and BG-HGNN expose that separate width. HGT, GTC and SeHGNN divide the hidden dimension across their heads. SPCA's Eq. 1 uses a single full-vector score, so its attention has no separate head-width axis.

The default attention head count is eight, and the final embedding width is always 128 regardless of internal search dimensions. A supplied `--set` fixes that parameter and removes it from the search. All remaining applicable axes are enumerated exhaustively; there is no random search or silent pruning. Source-specific settings are centralized in `model_defaults()` and recorded per trial. A winning dimension of 128 is an empirical manuscript result, not forced onto the hidden-dimension search.

Three tuning seeds are used for every configuration. Within each seed, training uses Adam and validation runs after every epoch, up to 500 epochs, stopping after 100 consecutive epochs without improvement. Higher validation Macro-F1 wins; lower validation loss breaks exact ties. Configurations maximize mean selected-checkpoint validation Macro-F1. Mean validation loss breaks configuration ties; first grid order breaks remaining exact ties. Five reporting seeds then train fresh models with the selected configuration. Test evaluation is separate from `fit()` and runs once at the selected checkpoint. Reports use sample standard deviation (`ddof=1`); one-run reports use `null` for the undefined sample deviation.

The manuscript specifies seed counts but not numeric seed lists. Defaults are tuning `[0, 1, 2]` and reporting `[42, 43, 44, 45, 46]`; both are configurable. `--phase report` without `--selected-config` runs the explicit/default settings without claiming a grid-search result. `--phase tune` does not compute test metrics.

## Splits and data

Main NC ratios (20%, 40%, 60%) use the **existing training pool within each class**. Low-label ratios (5%, 10%) use **all labeled nodes per class**, sampled outside fixed validation/test sets. Both select deterministic per-class prefixes, so larger budgets contain smaller budgets for the same seed. Counts use nearest-integer rounding with at least one training example per class; impossible budgets fail. Loader holdouts remain fixed across models and reporting seeds. Explicit holdout overrides use a fixed `--split-seed` independently of model/training-label seeds. Label-derived HetTree features use only the selected training labels and remove self-label return paths.

LastFM uses a 70/10/20 interaction split; MovieLens-1M uses 80/10/10. All observed target positives are collected before splitting. Held-out target interactions are removed in both directions before any encoder preparation. Negatives are sampled without replacement from pairs absent from the complete observed graph, one per positive, and are disjoint across the three splits. LP graph caches are built anew per split. The paper gives no separate LP checkpoint metric; the explicit implementation choice is validation ROC-AUC, then lower BCE, with the same patience and epoch limit. AP is also reported. This assumption is recorded in the run plan.

Original PyG/MAGNN DBLP and IMDB caches are supported. DBLP's raw-title fallback is detected by benchmark validation. The existing ACM raw loader has different statistics from Table 1; no arbitrary subsampling is used to make its counts look correct. Supply the matching graph via `--graph` or use the explicitly marked `--allow-data-mismatch` variant. All local graphs should contain `x`, target `y`, boolean `train_mask`, `val_mask`, `test_mask`, and typed PyG `edge_index` tensors. Files passed to `--graph` must be trusted because PyG graph loading uses Python object deserialization.

LastFM uses the MAGNN/PyG processed HetRec graph (1,892 users, 17,632 artists, 1,088 tags). PyG's supplied held-out positives are reunited with observed interactions before applying this protocol. The release has no features; fixed Gaussian features use a configurable width and feature seed, shared across models. The manuscript does not specify those input features, so this is a documented assumption rather than a verified historical choice. MovieLens-1M retains all 3,883 metadata movies, 6,040 users and 18 genres. Inputs use user demographic categories, movie genre indicators and genre identities, with no rating-derived features. These choices are recorded in the manifest.

K-means uses the selected NC embeddings, the known number of classes and the reporting seed, with `n_init=10`. It evaluates all target nodes by default; `--cluster-split test` restricts evaluation to held-out nodes. The manuscript does not specify the clustering subset or K-means restart count, so both choices are explicit.

## Baseline adaptations

Each model folder separates the runnable adaptation from an immutable, source-only upstream snapshot. The original preprocessing code is available in that snapshot. The shared runner ports the necessary preprocessing to PyG/SciPy so all models receive the same graph, masks and Table 3 paths; it does not invoke upstream dataset-specific training or evaluation scripts. These adaptations must not be described as bit-for-bit original runs.

- GCN/GAT use the union of induced endpoint graphs. RGCN, HGT, Simple-HGN and BG-HGNN retain one relation ID per selected metapath. Type-aware models therefore operate on one receiving node type per endpoint encoder under the manuscript's common-input adaptation.
- MAGNN retains ordered instance features and RotatE0 by default. The upstream DBLP sample cap is 100; this adapter samples up to 100 complete instances per receiver using prefix-count sampling with replacement when needed. All instances are available with `--set magnn.max_instances=null`. This sampling rule differs from the source's neighbor-based sampling and is recorded explicitly.
- HeCo retains metapath/schema views and its multi-positive contrastive loss. Its schema view receives the selected metapath relations, uses the common eight-head setting, and aggregates all supplied neighbors instead of CUDA-only per-type neighbor sampling. Semantic positives come from top-ranked selected-path neighbors plus self pairs.
- GTC retains Hop2Token, a Transformer per metapath, semantic fusion, a relation-wise GNN branch and co-contrastive loss. The original Transformer layers are retained in `transformer.py`; the adapter corrects the upstream constructor's exchanged head/layer arguments and handles singleton batches without dropping the batch dimension.
- For HeCo/GTC NC, contrastive representation learning and a detached linear evaluation head advance together each epoch, enabling the common validation selection rule. The encoder receives the contrastive loss; output/evaluation projections receive cross entropy. For LP, BCE and the contrastive objective train jointly. This evaluation cadence is a shared-protocol adaptation, not the original upstream post-hoc evaluator.
- SeHGNN keeps the existing feature-only adaptation: selected propagated channels plus raw target features, without label channels or multi-stage pseudo-labeling.
- HetTree's existing independent implementation is preserved. Fixed selected paths are completed with zero-input ancestor nodes, preserving tree connectivity without adding unselected semantic features. Training-label matching, subtree attention and feature/label residuals remain intact. For shared embeddings, only terminal output projections are sized to the requested embedding width. Its unavailable official source is not claimed to be bundled.
- BG-HGNN uses the source's random type/relation encodings and low-rank multiplicative fusion before a shared homogeneous GAT. Time-based reseeding and hardcoded CUDA placement are removed. Under a single-type symmetric endpoint graph, signed relation encodings can cancel; this follows the source calculation and is documented rather than replaced by another architecture.

The manuscript's random-walk defaults are 40 walks, length 100 and window 5. None of the runnable adapters here consumes random-walk embeddings; those settings are therefore not applied to unrelated operators. Available upstream preprocessing scripts remain in each source snapshot.

## Local output and verification

Every executed run writes a manifest with software versions, effective options, selected paths, node/edge counts, dataset discrepancies, and a SHA-256 graph/features/labels/splits fingerprint. Tuning records retain validation metrics only; reporting records add the one test evaluation. Checkpoint saving is opt-in. A new timestamped `local/experiments/` directory is used by default; an existing run manifest is never overwritten.

Unit tests cover SPCA's manual equation values, internal-only context, no intermediate feature propagation, shared sequence encoders, scalar-shift cancellation, graph composition, all model/dataset adapters, synthetic optimizer steps, LP splits/decoding, selection ties, sample standard deviation and no-data CLI dry runs. No benchmark performance or GPU memory claim follows from these checks.
