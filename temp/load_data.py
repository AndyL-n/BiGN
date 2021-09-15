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

def sparseFeature(feat, feat_num, embed_dim=8):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def load_data(file, embed_dim=32, path='Data/'):
    """
    :param file: A string. dataset name.
    :param embed_dim: A scalar. latent factor.
    :return: user_num, item_num, train_df, test_df
    """
    print('============Load Dateset====================')
    print('loading:\t' + file)
    train_file = 'Data/' + file + '/train.txt'
    test_file = 'Data/' + file + '/test.txt'
    train_dict, test_dict = defaultdict(list), defaultdict(list)
    user_num, item_num = 0, 0
    train_num, test_num = 0, 0

    with open(train_file) as f_train:
        for line in f_train.readlines():
            line = [int(i) for i in line.strip('\n').split(' ')]
            u_id = line[0]
            items = line[1:]
            train_num += len(items)
            train_dict[u_id] = items
            user_num = max(u_id, user_num)
            item_num = max(max(items), item_num)

    with open(test_file) as t_train:
        for line in t_train.readlines():
            line = line.strip('\n').split(' ')
            u_id = int(line[0])
            line = line[1:]
            if len(line) > 0 and line[0] != '':
                items = [int(i) for i in line]
            else:
                items = []
            test_num += len(items)
            test_dict[u_id] = items
            user_num = max(u_id, user_num)
            if items != []:
                item_num = max(max(items), item_num)
    user_num += 1
    item_num += 1
    print('user_num: ' + str(user_num) + ',item_num：' + str(item_num) + ',interactions：' + str(train_num + test_num))
    print('train_num：' + str(train_num) + ',test_num：' + str(test_num))

    print('============Create Norm Adj=================')


    R = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u_id in train_dict:
        for i_id in train_dict[u_id]:
            R[u_id, i_id] = 1
    print(R)
    #
    # R = R.tocsr()
    # user_sim = cosine_similarity(R)
    # print(user_sim[0].argsort()[::-1])
    # R = R.T
    # # print(R.shape)
    # item_sim = cosine_similarity(R)
    # print(item_sim[10000].argsort()[-1])
    # print(item_sim[40979].argsort()[::-1])
    # print(item_sim[40980].argsort()[::-1])
    # R = R.T
    # def create_adj_mat(R, n_users, n_items):
    #     t1 = time()
    #     adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    #     adj_mat = adj_mat.tolil()
    #     R = R.tolil()
    #     print(R.shape)
    #     # prevent memory from overflowing
    #     for i in range(5):
    #         adj_mat[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5), n_users:] = \
    #             R[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)]
    #         adj_mat[n_users:, int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)] = \
    #             R[int(n_users * i / 5.0):int(n_users * (i + 1.0) / 5)].T
    #     adj_mat = adj_mat.todok()
    #     print('already create adjacency matrix', adj_mat.shape, time() - t1)
    #
    #     t2 = time()
    #
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
    adj_mat = sp.load_npz('Data/' + file + '/adj_mat.npz')
    norm_adj_mat = sp.load_npz('Data/' + file + '/adj_norm_mat.npz')
    mean_adj_mat = sp.load_npz('Data/' + file + '/adj_mean_mat.npz')
    pre_adj_mat = sp.load_npz('Data/' + file + '/adj_pre_mat.npz')











    # print('============Create Test Data================')
    feat_col = [sparseFeature('user_id', user_num, embed_dim), sparseFeature('item_id', item_num, embed_dim)]
    # # shuffle 随机排序列表
    # train_dict = train_dict.sample(frac=1).reset_index(drop=True)
    # val_dict = val_dict.sample(frac=1).reset_index(drop=True)
    # all_user = np.array([i for i in range(user_num)], dtype=int)
    # all_item = np.array([i for i in range(item_num)], dtype=int)
    #
    # train = [np.array(train_dict['user']), np.array(train_dict['pos']), np.array(train_dict['neg'])]
    #          # np.array([all_user for _ in range(len(train_dict['user']))]), \
    #          # np.array([all_item for _ in range(len(train_dict['user']))])]
    # val = [np.array(val_dict['user']), np.array(val_dict['pos']), np.array(val_dict['neg'])]
    #        # np.array([all_user for _ in range(len(val_dict['user']))]), \
    #        # np.array([all_item for _ in range(len(val_dict['user']))])]
    #
    # test_data = defaultdict(list)
    # with open(test_file) as f_test:
    #     for line in tqdm(f_test.readlines()):
    #         line = [int(x) for x in line.strip('\n').split(' ')[:101]]
    #         test_data['user'].append(line[0])
    #         test_data['pos'].append(line[1])
    #         test_data['neg'].append(np.array(line[2:]))
    #
    # test = [np.array(test_data['user']), np.array(test_data['pos']), np.array(test_data['neg'])]
    print('============Data Preprocess End=============')
    #
    return feat_col, norm_adj_mat
    # return feat_col, train, val ,test


files = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m', 'ml-100k', 'yelp2018']
for file in files:
    feat_col, norm_adj_mat = load_data(file)
    # print(train)

