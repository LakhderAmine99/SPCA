# Source cells extracted from preprocess_LastFM.ipynb
import pathlib
import pickle

import numpy as np
import scipy.sparse
import scipy.io
import pandas as pd

# %%

save_prefix = 'data/preprocessed/LastFM_processed/'

# %%

user_artist = pd.read_csv('data/raw/LastFM/user_artist.dat', encoding='utf-8', delimiter='\t', names=['userID', 'artistID', 'weight'])
user_friend = pd.read_csv('data/raw/LastFM/user_user(original).dat', encoding='utf-8', delimiter='\t', names=['userID', 'friendID'])
artist_tag = pd.read_csv('data/raw/LastFM/artist_tag.dat', encoding='utf-8', delimiter='\t', names=['artistID', 'tagID'])
num_user = 1892
num_artist = 17632
num_tag = 11945

# %%

train_val_test_idx = np.load('data/raw/LastFM/train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']
 
user_artist = user_artist.loc[train_idx].reset_index(drop=True)

# %%

# build the adjacency matrix
# 0 for user, 1 for artist, 2 for tag
dim = num_user + num_artist + num_tag

type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

adjM = np.zeros((dim, dim), dtype=int)
for _, row in user_artist.iterrows():
    uid = row['userID'] - 1
    aid = num_user + row['artistID'] - 1
    adjM[uid, aid] = max(1, row['weight'])
    adjM[aid, uid] = max(1, row['weight'])
for _, row in user_friend.iterrows():
    uid = row['userID'] - 1
    fid = row['friendID'] - 1
    adjM[uid, fid] = 1
for _, row in artist_tag.iterrows():
    aid = num_user + row['artistID'] - 1
    tid = num_user + num_artist + row['tagID'] - 1
    adjM[aid, tid] += 1
    adjM[tid, aid] += 1

# %%

# filter out artist-tag links with counts less than 2
adjM[num_user:num_user+num_artist, num_user+num_artist:] = adjM[num_user:num_user+num_artist, num_user+num_artist:] * (adjM[num_user:num_user+num_artist, num_user+num_artist:] > 1)
adjM[num_user+num_artist:, num_user:num_user+num_artist] = np.transpose(adjM[num_user:num_user+num_artist, num_user+num_artist:])

valid_tag_idx = adjM[num_user:num_user+num_artist, num_user+num_artist:].sum(axis=0).nonzero()[0]
num_tag = len(valid_tag_idx)
dim = num_user + num_artist + num_tag
type_mask = np.zeros((dim), dtype=int)
type_mask[num_user:num_user+num_artist] = 1
type_mask[num_user+num_artist:] = 2

adjM_reduced = np.zeros((dim, dim), dtype=int)
adjM_reduced[:num_user+num_artist, :num_user+num_artist] = adjM[:num_user+num_artist, :num_user+num_artist]
adjM_reduced[num_user:num_user+num_artist, num_user+num_artist:] = adjM[num_user:num_user+num_artist, num_user+num_artist:][:, valid_tag_idx]
adjM_reduced[num_user+num_artist:, num_user:num_user+num_artist] = np.transpose(adjM_reduced[num_user:num_user+num_artist, num_user+num_artist:])

adjM = adjM_reduced

# %%

user_artist_list = {i: adjM[i, num_user:num_user+num_artist].nonzero()[0] for i in range(num_user)}
artist_user_list = {i: adjM[num_user + i, :num_user].nonzero()[0] for i in range(num_artist)}
user_user_list = {i: adjM[i, :num_user].nonzero()[0] for i in range(num_user)}
artist_tag_list = {i: adjM[num_user + i, num_user+num_artist:].nonzero()[0] for i in range(num_artist)}
tag_artist_list = {i: adjM[num_user + num_artist + i, num_user:num_user+num_artist].nonzero()[0] for i in range(num_tag)}

# %%

# 0-1-0
u_a_u = []
for a, u_list in artist_user_list.items():
    u_a_u.extend([(u1, a, u2) for u1 in u_list for u2 in u_list])
u_a_u = np.array(u_a_u)
u_a_u[:, 1] += num_user
sorted_index = sorted(list(range(len(u_a_u))), key=lambda i : u_a_u[i, [0, 2, 1]].tolist())
u_a_u = u_a_u[sorted_index]

# 1-2-1
a_t_a = []
for t, a_list in tag_artist_list.items():
    a_t_a.extend([(a1, t, a2) for a1 in a_list for a2 in a_list])
a_t_a = np.array(a_t_a)
a_t_a += num_user
a_t_a[:, 1] += num_artist
sorted_index = sorted(list(range(len(a_t_a))), key=lambda i : a_t_a[i, [0, 2, 1]].tolist())
a_t_a = a_t_a[sorted_index]

# 0-1-2-1-0
u_a_t_a_u = []
for a1, t, a2 in a_t_a:
    if len(artist_user_list[a1 - num_user]) == 0 or len(artist_user_list[a2 - num_user]) == 0:
        continue
    candidate_u1_list = np.random.choice(len(artist_user_list[a1 - num_user]), int(0.2 * len(artist_user_list[a1 - num_user])), replace=False)
    candidate_u1_list = artist_user_list[a1 - num_user][candidate_u1_list]
    candidate_u2_list = np.random.choice(len(artist_user_list[a2 - num_user]), int(0.2 * len(artist_user_list[a2 - num_user])), replace=False)
    candidate_u2_list = artist_user_list[a2 - num_user][candidate_u2_list]
    u_a_t_a_u.extend([(u1, a1, t, a2, u2) for u1 in candidate_u1_list for u2 in candidate_u2_list])
u_a_t_a_u = np.array(u_a_t_a_u)
sorted_index = sorted(list(range(len(u_a_t_a_u))), key=lambda i : u_a_t_a_u[i, [0, 4, 1, 2, 3]].tolist())
u_a_t_a_u = u_a_t_a_u[sorted_index]

# 0-0
u_u = user_friend.to_numpy(dtype=np.int32) - 1
sorted_index = sorted(list(range(len(u_u))), key=lambda i : u_u[i].tolist())
u_u = u_u[sorted_index]

# 1-0-1
a_u_a = []
for u, a_list in user_artist_list.items():
    a_u_a.extend([(a1, u, a2) for a1 in a_list for a2 in a_list])
a_u_a = np.array(a_u_a)
a_u_a[:, [0, 2]] += num_user
sorted_index = sorted(list(range(len(a_u_a))), key=lambda i : a_u_a[i, [0, 2, 1]].tolist())
a_u_a = a_u_a[sorted_index]

# 1-0-0-1
a_u_u_a = []
for u1, u2 in u_u:
    a_u_u_a.extend([(a1, u1, u2, a2) for a1 in user_artist_list[u1] for a2 in user_artist_list[u2]])
a_u_u_a = np.array(a_u_u_a)
a_u_u_a[:, [0, 3]] += num_user
sorted_index = sorted(list(range(len(a_u_u_a))), key=lambda i : a_u_u_a[i, [0, 3, 1, 2]].tolist())
a_u_u_a = a_u_u_a[sorted_index]

# %%

expected_metapaths = [
    [(0, 1, 0), (0, 1, 2, 1, 0), (0, 0)],
    [(1, 0, 1), (1, 2, 1), (1, 0, 0, 1)]
]
# create the directories if they do not exist
for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): u_a_u,
                            (0, 1, 2, 1, 0): u_a_t_a_u,
                            (0, 0): u_u,
                            (1, 0, 1): a_u_a,
                            (1, 2, 1): a_t_a,
                            (1, 0, 0, 1): a_u_u_a}

# write all things
target_idx_lists = [np.arange(num_user), np.arange(num_artist)]
offset_list = [0, num_user]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]
        
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        #np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)
        
        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right

scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
np.save(save_prefix + 'node_types.npy', type_mask)

# %%

# output user_artist.npy
user_artist = pd.read_csv('data/raw/LastFM/user_artist.dat', encoding='utf-8', delimiter='\t', names=['userID', 'artistID', 'weight'])
user_artist = user_artist[['userID', 'artistID']].to_numpy()
user_artist = user_artist - 1
np.save(save_prefix + 'user_artist.npy', user_artist)

# %%

# output positive and negative samples for training, validation and testing

np.random.seed(453289)
save_prefix = 'data/preprocessed/LastFM_processed/'
num_user = 1892
num_artist = 17632
user_artist = np.load('data/preprocessed/LastFM_processed/user_artist.npy')
train_val_test_idx = np.load('data/raw/LastFM/train_val_test_idx.npz')
train_idx = train_val_test_idx['train_idx']
val_idx = train_val_test_idx['val_idx']
test_idx = train_val_test_idx['test_idx']

neg_candidates = []
counter = 0
for i in range(num_user):
    for j in range(num_artist):
        if counter < len(user_artist):
            if i == user_artist[counter, 0] and j == user_artist[counter, 1]:
                counter += 1
            else:
                neg_candidates.append([i, j])
        else:
            neg_candidates.append([i, j])
neg_candidates = np.array(neg_candidates)

idx = np.random.choice(len(neg_candidates), len(val_idx) + len(test_idx), replace=False)
val_neg_candidates = neg_candidates[sorted(idx[:len(val_idx)])]
test_neg_candidates = neg_candidates[sorted(idx[len(val_idx):])]

train_user_artist = user_artist[train_idx]
train_neg_candidates = []
counter = 0
for i in range(num_user):
    for j in range(num_artist):
        if counter < len(train_user_artist):
            if i == train_user_artist[counter, 0] and j == train_user_artist[counter, 1]:
                counter += 1
            else:
                train_neg_candidates.append([i, j])
        else:
            train_neg_candidates.append([i, j])
train_neg_candidates = np.array(train_neg_candidates)

np.savez(save_prefix + 'train_val_test_neg_user_artist.npz',
         train_neg_user_artist=train_neg_candidates,
         val_neg_user_artist=val_neg_candidates,
         test_neg_user_artist=test_neg_candidates)
np.savez(save_prefix + 'train_val_test_pos_user_artist.npz',
         train_pos_user_artist=user_artist[train_idx],
         val_pos_user_artist=user_artist[val_idx],
         test_pos_user_artist=user_artist[test_idx])
