# Source cells extracted from preprocess_DBLP.ipynb
import pathlib

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import utils.preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

from utils.data import load_glove_vectors

# %%

save_prefix = 'data/preprocessed/DBLP_processed/'
num_ntypes = 4

# %%

author_label = pd.read_csv('data/raw/DBLP/author_label.txt', sep='\t', header=None, names=['author_id', 'label', 'author_name'], keep_default_na=False, encoding='utf-8')
paper_author = pd.read_csv('data/raw/DBLP/paper_author.txt', sep='\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
paper_conf = pd.read_csv('data/raw/DBLP/paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
paper_term = pd.read_csv('data/raw/DBLP/paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')
papers = pd.read_csv('data/raw/DBLP/paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title'], keep_default_na=False, encoding='cp1252')
terms = pd.read_csv('data/raw/DBLP/term.txt', sep='\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
confs = pd.read_csv('data/raw/DBLP/conf.txt', sep='\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')

glove_dim = 50
glove_vectors = load_glove_vectors(dim=glove_dim)

# %%

# filter out all nodes which does not associated with labeled authors
labeled_authors = author_label['author_id'].to_list()
paper_author = paper_author[paper_author['author_id'].isin(labeled_authors)].reset_index(drop=True)
valid_papers = paper_author['paper_id'].unique()
papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
valid_terms = paper_term['term_id'].unique()
terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)

# %%

# term lemmatization and grouping
lemmatizer = WordNetLemmatizer()
lemma_id_mapping = {}
lemma_list = []
lemma_id_list = []
i = 0
for _, row in terms.iterrows():
    i += 1
    lemma = lemmatizer.lemmatize(row['term'])
    lemma_list.append(lemma)
    if lemma not in lemma_id_mapping:
        lemma_id_mapping[lemma] = row['term_id']
    lemma_id_list.append(lemma_id_mapping[lemma])
terms['lemma'] = lemma_list
terms['lemma_id'] = lemma_id_list

term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
lemma_id_list = []
for _, row in paper_term.iterrows():
    lemma_id_list.append(term_lemma_mapping[row['term_id']])
paper_term['lemma_id'] = lemma_id_list

paper_term = paper_term[['paper_id', 'lemma_id']]
paper_term.columns = ['paper_id', 'term_id']
paper_term = paper_term.drop_duplicates()
terms = terms[['lemma_id', 'lemma']]
terms.columns = ['term_id', 'term']
terms = terms.drop_duplicates()

# %%

# filter out stopwords from terms
stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
stopword_id_list = terms[terms['term'].isin(stopwords)]['term_id'].to_list()
paper_term = paper_term[~(paper_term['term_id'].isin(stopword_id_list))].reset_index(drop=True)
terms = terms[~(terms['term'].isin(stopwords))].reset_index(drop=True)

# %%

# remove terms not found in GloVe
#terms_not_in_glove = []
#for _, row in terms.iterrows():
#    if row['term'] not in glove_vectors:
#        terms_not_in_glove.append(row['term'])
#term_ids_not_in_glove = terms[terms['term'].isin(terms_not_in_glove)]['term_id'].to_list()
#terms = terms[~(terms['term'].isin(terms_not_in_glove))].reset_index(drop=True)
#paper_term = paper_term[~(paper_term['term_id'].isin(term_ids_not_in_glove))].reset_index(drop=True)

# %%

# consider only terms associated with at least two papers
# if having meaningful word vectors for terms (e.g. GloVe)?

# %%

author_label = author_label.sort_values('author_id').reset_index(drop=True)
papers = papers.sort_values('paper_id').reset_index(drop=True)
terms = terms.sort_values('term_id').reset_index(drop=True)
confs = confs.sort_values('conf_id').reset_index(drop=True)

# %%

# extract labels of authors
labels = author_label['label'].to_numpy()

# %%

# build the adjacency matrix for the graph consisting of authors, papers, terms and conferences
# 0 for authors, 1 for papers, 2 for terms, 3 for conferences
dim = len(author_label) + len(papers) + len(terms) + len(confs)
type_mask = np.zeros((dim), dtype=int)
type_mask[len(author_label):len(author_label)+len(papers)] = 1
type_mask[len(author_label)+len(papers):len(author_label)+len(papers)+len(terms)] = 2
type_mask[len(author_label)+len(papers)+len(terms):] = 3

author_id_mapping = {row['author_id']: i for i, row in author_label.iterrows()}
paper_id_mapping = {row['paper_id']: i + len(author_label) for i, row in papers.iterrows()}
term_id_mapping = {row['term_id']: i + len(author_label) + len(papers) for i, row in terms.iterrows()}
conf_id_mapping = {row['conf_id']: i + len(author_label) + len(papers) + len(terms) for i, row in confs.iterrows()}

adjM = np.zeros((dim, dim), dtype=int)
for _, row in paper_author.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = author_id_mapping[row['author_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_term.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = term_id_mapping[row['term_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_conf.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = conf_id_mapping[row['conf_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1

# %%

# use HAN paper's preprocessed data as the features of authors (https://github.com/Jhy1993/HAN)
mat = scipy.io.loadmat('data/raw/DBLP/DBLP4057_GAT_with_idx.mat')
features_author = np.array(list(zip(*sorted(zip(labeled_authors, mat['features']), key=lambda tup: tup[0])))[1])
features_author = scipy.sparse.csr_matrix(features_author)

# %%

# use bag-of-words representation of paper titles as the features of papers
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
vectorizer = CountVectorizer(min_df=2, stop_words=stopwords, tokenizer=LemmaTokenizer())
features_paper = vectorizer.fit_transform(papers['paper_title'].values)

# %%

# use pretrained GloVe vectors as the features of terms
features_term = np.zeros((len(terms), glove_dim))
for i, row in terms.iterrows():
    features_term[i] = glove_vectors.get(row['term'], glove_vectors['the'])

# %%

expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 3, 1)],
    [(2, 1, 2), (2, 1, 0, 1, 2), (2, 1, 3, 1, 2)],
    [(3, 1, 3), (3, 1, 0, 1, 3), (3, 1, 2, 1, 3)]
]
# create the directories if they do not exist
for i in range(1):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
for i in range(1):
    # get metapath based neighbor pairs
    neighbor_pairs = utils.preprocess.get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
    # construct and save metapath-based networks
    G_list = utils.preprocess.get_networkx_graph(neighbor_pairs, type_mask, i)
    
    # save data
    # networkx graph (metapath specific)
    for G, metapath in zip(G_list, expected_metapaths[i]):
        nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
    # node indices of edge metapaths
    all_edge_metapath_idx_array = utils.preprocess.get_edge_metapath_idx_array(neighbor_pairs)
    for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
        np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)
# save data
# all nodes adjacency matrix
scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
# all nodes (authors, papers, terms and conferences) features
# currently only have features of authors, papers and terms
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)
scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)
np.save(save_prefix + 'features_{}.npy'.format(2), features_term)
# all nodes (authors, papers, terms and conferences) type labels
np.save(save_prefix + 'node_types.npy', type_mask)
# author labels
np.save(save_prefix + 'labels.npy', labels)
# author train/validation/test splits
rand_seed = 1566911444
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
train_idx, test_idx = train_test_split(train_idx, test_size=3257, random_state=rand_seed)
train_idx.sort()
val_idx.sort()
test_idx.sort()
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx=val_idx,
         train_idx=train_idx,
         test_idx=test_idx)

# %%

# post-processing for mini-batched training
target_idx_list = np.arange(4057)
for metapath in [(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)]:
    edge_metapath_idx_array = np.load(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.npy')
    target_metapaths_mapping = {}
    for target_idx in target_idx_list:
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[edge_metapath_idx_array[:, 0] == target_idx][:, ::-1]
    out_file = open(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb')
    pickle.dump(target_metapaths_mapping, out_file)
    out_file.close()
