# HetTree

Paper: [Heterogeneous Graph Neural Network on Semantic Tree](https://doi.org/10.1609/aaai.v39i16.33860), Guan et al., AAAI 2025. [Author-hosted paper](https://taesoo.kim/pubs/2025/guan:hettree.pdf). Source link from the paper: [microsoft/HetTree](https://github.com/microsoft/HetTree).

`model.py` contains the architecture, `layers.py` provides the reusable tree layers, and `preprocessing.py` prepares propagated features and training labels.

The model matches propagated features and training labels (Eq. 3), forms a reference from parent and child features, applies attention over encoded child subtrees (Eqs. 4–5), and sums tree representations with feature/label residuals (Eq. 6). Path keys use source-to-target order, with each tree parent represented by a suffix.

The shared runner selects features using Table 3 paths, inserts zero-input ancestors, and propagates selected training labels with self-return paths removed. Terminal projections use the configured embedding width. LP uses feature inputs.

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

See `tests/test_hettree.py`, `tests/test_paper_models.py` and [experiment settings](../../../../docs/paper_protocol.md).
