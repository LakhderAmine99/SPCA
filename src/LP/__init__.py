"""Link-prediction adapters for the shared model library."""
from .models import LinkPredictor, build_model
from .decoders import DotProductDecoder, MLPDecoder

__all__ = ['LinkPredictor', 'build_model', 'DotProductDecoder', 'MLPDecoder']
