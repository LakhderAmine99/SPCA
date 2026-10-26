"""Official MAGNN process integration; retains metapath-instance preprocessing.

Example: python -m src.model.baselines.magnn --repo third_party/MAGNN
         --entrypoint run_IMDB.py -- --help
"""
from ._upstream import UpstreamBaseline, main

MAGNN = UpstreamBaseline('MAGNN', 'https://github.com/cynricfu/MAGNN',
                         entrypoint='run_IMDB.py')

if __name__ == '__main__':
    main(MAGNN)
