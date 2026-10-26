"""HGB is the benchmark; its Simple-HGN baseline is exposed here as an alias."""
from .simplehgn import SimpleHGN

HGB = SimpleHGN
__all__ = ['HGB', 'SimpleHGN']
