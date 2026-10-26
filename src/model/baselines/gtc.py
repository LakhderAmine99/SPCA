"""Official GTC process integration, including Hop2Token and co-contrastive loss."""
from ._upstream import UpstreamBaseline, main

GTC = UpstreamBaseline('GTC', 'https://github.com/PHD-lanyu/GTC',
                      workdir='code', entrypoint='main.py')

if __name__ == '__main__':
    main(GTC)
