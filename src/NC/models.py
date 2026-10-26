"""NC outputs are raw node logits; use cross entropy on the training mask."""
from torch import nn
from src.model.tasks import build_backbone


class NodeClassifier(nn.Module):
    """Adapt an embedding encoder to NC without changing its input contract."""

    def __init__(self, encoder, embedding_channels, num_classes):
        super().__init__()
        if embedding_channels < 1 or num_classes < 1:
            raise ValueError('Embedding and class counts must be positive')
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_channels, num_classes)

    def forward(self, *args, **kwargs):
        return self.classifier(self.encoder(*args, **kwargs))


def build_model(name, num_classes, **model_kwargs):
    """Build SPCA or a native baseline with its existing classification head.

    Other arguments and forward inputs follow the shared architecture. This
    avoids placing a second classifier on top of existing class scores.
    """
    return build_backbone(name, num_classes, **model_kwargs)
