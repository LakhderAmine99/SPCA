"""SPCA model family."""
from .encoders import (
    SimpleMetapathConv,
    SPCAConv,
    SPCAConv2,
    RegularizedSPCAConv,
    SPCAConv3,
    StackedSPCA,
    SubpathClassifier,
)
from .model import SPCA
from .layers import SPCALayer, DotProductConv, SubpathContext

__all__ = ['SPCALayer', 'DotProductConv', 'SubpathContext', 'SimpleMetapathConv', 'SPCAConv', 'SPCAConv2', 'RegularizedSPCAConv', 'SPCAConv3', 'StackedSPCA', 'SubpathClassifier', 'SPCA']
