from scipy.sparse import csr_matrix
import numpy as np
from math import sqrt, log
from sys import exit
from tqdm import tqdm
from heapq import nlargest
users = []
idx = [[],[]]
n_user, n_item, n_data = 0, 0, 0
data = 'Data/gowalla/train.txt'
with open(data) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            users.append(uid)
            idx[0].extend([uid] * len(items))
            idx[1].extend(items)
            n_item = max(n_item, max(items))
            n_user = max(n_user, uid)
            n_data += len(items)

n_user, n_item = n_user + 1, n_item + 1
print(max(idx[0]), max(idx[1]))
print(n_user, n_item, n_data)
R = csr_matrix((np.ones(n_data), (idx[0], idx[1])), shape=(n_user, n_item))

users_D = np.array(R.sum(axis=1)).squeeze()
items_D = np.array(R.sum(axis=0)).squeeze()
idx[0], idx[1] = [],[]
T = R.transpose()
for i in range(n_item):
    scores = []
    for j in tqdm(range(n_item)):
        sim = 1 / sqrt(items_D[i] * items_D[j])
        score = 0
        for k in np.intersect1d(T[i].indices, T[j].indices):
            score += 1 / log(users_D[k] + 1)
        score = score * sim
        scores.append(score)
    print(scores)
    scores = np.array(scores)
    idx[0].extend([i] * 31)
    idx[1].extend(scores.argsort()[::-1][:31])
    print(idx[0] + [n_user] * len(idx[0]))
    print(idx[1])
    exit()

for i in range(n_user):
    scores = []
    for j in range(n_user):
        sim = 1 / sqrt(users_D[i] * users_D[j])
        score = 0
        for k in np.intersect1d(R[i].indices, R[j].indices):
            score += 1 / log(items_D[k] + 1)
        score = score * sim
        scores.append(score)
    print(scores)
    scores = np.array(scores)
    idx[0].extend([i] * 31)
    idx[1].extend(scores.argsort()[::-1][:31])
    print(idx[0] + [n_user] * len(idx[0]))
    print(idx[1])
    exit()
