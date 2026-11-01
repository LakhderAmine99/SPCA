# HetTree

Paper: [Heterogeneous Graph Neural Network on Semantic Tree](https://doi.org/10.1609/aaai.v39i16.33860), Guan et al., AAAI 2025. [Author-hosted paper](https://taesoo.kim/pubs/2025/guan:hettree.pdf). Original source URL: [microsoft/HetTree](https://github.com/microsoft/HetTree), unavailable when checked.

The existing local independent implementation is preserved. It is reorganized into `model.py`, `layers.py`, and `preprocessing.py` without changing its architecture. It is not labeled an identical copy of unavailable official code.

The paper comparison confirms matched propagated features/training labels (Eq. 3), a reference formed from parent and child features, attention over encoded child subtrees (Eqs. 4–5), and summed tree representations with feature/label residuals (Eq. 6). Source-to-target path keys make tree parents suffixes, reversing the paper's receiving-node-first notation.

The shared runner restricts features to Table 3 paths, inserts zero-input ancestors, and excludes held-out/self-return labels. It adapts terminal projection widths for shared embeddings; LP disables labels. These are common-protocol adaptations, not upstream equivalence claims.

```python
from src.model.baselines.hettree import HetTree, precompute_hettree_inputs
features, labels = precompute_hettree_inputs(
    x_dict, edge_index_dict, target_type, num_hops,
    labels=y, train_mask=train_mask, num_classes=num_classes,
    metapaths=selected_paths, include_ancestor_features=False,
)
model = HetTree(features, num_classes, label_features=labels)
logits = model()
```

See `tests/test_hettree.py`, `tests/test_paper_models.py` and [protocol notes](../../../../docs/paper_protocol.md). No official snapshot or pretrained parameters are invented or bundled.
