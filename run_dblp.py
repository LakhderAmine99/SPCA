"""DBLP: all manuscript models, grid search, then five reporting runs."""
from src.experiments.runner import main
if __name__ == '__main__':
    main(dataset='dblp')
