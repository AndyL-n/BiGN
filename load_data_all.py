"""
author: L
date: 2021/8/25 14:06
"""

import numpy as np
import scipy.sparse as sp
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from time import time

def sparseFeature(feat, feat_num, embed_dim=32):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def load_data(file, embed_dim = 8, test_neg_num=100):
    """
    :param file: A string. dataset name.
    :param embed_dim: A scalar. latent factor.
    :return: user_num, item_num, train_df, test_df
    """
    print('============Load Dateset====================')
    print('loading:\t' + file)
    train_file = 'Data/' + file + '/train.txt'
    test_file = 'Data/' + file + '/test.txt'
    user_num, item_num = 0, 0
    n_train, n_test = 0, 0
    train_dict, test_dict = dict(), dict()
    exist_items = []
    with open(train_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                items = [int(i) for i in line[1:]]
                item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                train_dict[u_id] = items
                n_train += len(items)
                exist_items += items
        f.close()

    with open(test_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                line = line[1:]
                if len(line) > 0 and line[0] != '':
                    items = [int(i) for i in line]
                    item_num = max(item_num, max(items))
                else:
                    items = []
                user_num = max(user_num, u_id)
                test_dict[u_id] = items
                n_test += len(items)
                exist_items += items
        f.close()

    user_num += 1
    item_num += 1
    feat_col = [sparseFeature('user_id', user_num, embed_dim), sparseFeature('item_id', item_num, embed_dim)]

    print(user_num,item_num)
    print(n_train, n_test)

    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)

    for u_id in tqdm(train_dict):
        pos_list = [int(i) for i in train_dict[u_id]]
        neg_list = list(set(exist_items) - set(pos_list))
        chose = random.randint(0,len(pos_list) - 1)
        for index, pos in enumerate(pos_list):
            if(index == chose):
                val_data['user_id'].append(u_id)
                val_data['pos_id'].append(pos)
                val_data['neg_id'].append(neg_list[chose])
            else:
                train_data['user_id'].append(u_id)
                train_data['pos_id'].append(pos)
                train_data['neg_id'].append(neg_list[chose])


    print('============Create Norm Adj=================')
    adj_mat = sp.load_npz('Data/' + file + '/adj_mat.npz')
    norm_adj_mat = sp.load_npz('Data/' + file + '/adj_norm_mat.npz')
    mean_adj_mat = sp.load_npz('Data/' + file + '/adj_mean_mat.npz')
    pre_adj_mat = sp.load_npz('Data/' + file + '/adj_pre_mat.npz')
    # R = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    # with open(train_file) as f_train:
    #     for line in tqdm(f_train.readlines()):
    #         line = [int(x) for x in line.strip('\n').split(' ')]
    #         R[line[0], line[1]] = 1
    #
    # # R = R.tocsr()
    # # user_sim = sp.dok_matrix((user_num, user_num), dtype=np.float32)
    # # user_sim[:, :] = cosine_similarity(R)[:, :]
    # # R = R.T
    # # # # print(R.shape)
    # # item_sim = sp.dok_matrix((item_num, item_num), dtype=np.float32)
    # # item_sim[:, :] = cosine_similarity(R)[:, :]
    # # R = R.T
    # # print(item_sim[10000].argsort()[-1])
    # # print(item_sim[40979].argsort()[::-1])
    # # print(item_sim[40980].argsort()[::-1])
    # # R = R.T
    # # def create_adj_mat(R, n_users, n_items):
    # #     t1 = time()
    # #     adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    # #     adj_mat = adj_mat.tolil()
    # #     R = R.tolil()
    # #     print(R.shape)
    # #     # prevent memory from overflowing
    # #     for i in range(5):
    # #         adj_mat[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5), n_users:] = \
    # #             R[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)]
    # #         adj_mat[n_users:, int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)] = \
    # #             R[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)].T
    # #     adj_mat = adj_mat.todok()
    # #     print('already create adjacency matrix', adj_mat.shape, time() - t1)
    # #
    # #     t2 = time()
    # #
    #     def normalized_adj_single(adj):
    #         rowsum = np.array(adj.sum(1))
    #         d_inv = np.power(rowsum, -1).flatten()
    #         d_inv[np.isinf(d_inv)] = 0.
    #         d_mat_inv = sp.diags(d_inv)
    #
    #         norm_adj = d_mat_inv.dot(adj)
    #         print('generate single-normalized adjacency matrix.')
    #         return norm_adj.tocoo()
    #
    #     def check_adj_if_equal(adj):
    #         dense_A = np.array(adj.todense())
    #         degree = np.sum(dense_A, axis=1, keepdims=False)
    #
    #         temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
    #         print('check normalized adjacency matrix whether equal to this laplacian matrix.')
    #         return temp
    #
    #     print(adj_mat.shape)
    #     norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    #     mean_adj_mat = normalized_adj_single(adj_mat)
    #
    #     print('already normalize adjacency matrix', time() - t2)
    #     return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
    #
    # adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(R, user_num, item_num)
    # sp.save_npz('Data/' + file + '/adj_mat.npz', adj_mat)
    # sp.save_npz('Data/' + file + '/adj_norm_mat.npz', norm_adj_mat)
    # sp.save_npz('Data/' + file + '/adj_mean_mat.npz', mean_adj_mat)
    # adj_mat = adj_mat
    # rowsum = np.array(adj_mat.sum(1))
    # d_inv = np.power(rowsum, -0.5).flatten()
    # d_inv[np.isinf(d_inv)] = 0.
    # d_mat_inv = sp.diags(d_inv)
    # norm_adj = d_mat_inv.dot(adj_mat)
    # norm_adj = norm_adj.dot(d_mat_inv)
    # print('generate pre adjacency matrix.')
    # pre_adj_mat = norm_adj.tocsr()
    # sp.save_npz('Data/' + file + '/adj_pre_mat.npz', norm_adj)

    print('============Create Test Dateset=============')

    for u_id in tqdm(test_dict):
        test_data['user_id'].append(u_id)
        test_data['pre_id'].append(np.array([int(i) for i in test_dict[u_id]]))
        test_data['true_id'].append(np.array([int(i) for i in train_dict[u_id]]))


    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']), np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']), np.array(val_data['neg_id'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pre_id']),np.array(test_data['true_id'])]

    print('============Load Dateset Success===========')
    return feat_col, train, val, test, norm_adj_mat

# load_data('yelp2018')