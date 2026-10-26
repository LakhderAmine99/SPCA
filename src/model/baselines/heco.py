"""Official HeCo process integration, including contrastive training/evaluation."""
from ._upstream import UpstreamBaseline, main

HeCo = UpstreamBaseline('HeCo', 'https://github.com/liun-online/HeCo',
                       workdir='code', entrypoint='main.py')

if __name__ == '__main__':
    main(HeCo)
