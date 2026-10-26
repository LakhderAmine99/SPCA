"""BG-HGNN external integration contract.

No verified upstream repository/entrypoint is bundled. Supply a verified local
implementation and explicit entrypoint; this module does not claim to implement
the paper's fusion/pretraining pipeline.
"""
from ._upstream import UpstreamBaseline, main

BGHGNN = UpstreamBaseline('BG-HGNN', 'https://arxiv.org/abs/2403.08207')

if __name__ == '__main__':
    main(BGHGNN)
