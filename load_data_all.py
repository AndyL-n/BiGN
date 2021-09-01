"""
author: L
date: 2021/8/25 14:06
"""

import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

def sparseFeature(feat, feat_num, embed_dim=8):
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
    test_file = 'Data/' + file + '/pinterest.test.txt'
    user_num, item_num = 0, 0
    n_train, n_test = 0, 0
    pos_dict, test_dict = dict(), dict()
    exist_items = []
    with open(train_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                items = [int(i) for i in line[1:]]
                item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                pos_dict[u_id] = items
                n_train += len(items)
                exist_items += items
        f.close()

    with open(test_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                items = [int(i) for i in line[1:]]
                item_num = max(item_num, max(items))
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
    train_data, val_data = defaultdict(list), defaultdict(list)

    for u_id in tqdm(pos_dict):
        pos_list = [int(i) for i in pos_dict[u_id]]
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

    print('============Create Test Dateset=============')
    def gen_neg(pos_list, exist_items, item):
        neg = pos_list[0]
        while neg in set(pos_list) or neg == item:
            chose = random.randint(0, len(exist_items) - 1)
            neg = exist_items[chose]
        return neg

    test_data = [[],[],[]]
    for u_id in tqdm(test_dict):
        test_list = [int(i) for i in test_dict[u_id]]
        pos_list = [int(i) for i in pos_dict[u_id]]
        for item in test_list:
            test_data[0].append(u_id)
            test_data[1].append(item)
            test_data[2].append(np.array([gen_neg(pos_list, exist_items, item) for _ in range(test_neg_num)]))


    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']), np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']), np.array(val_data['neg_id'])]
    test = [np.array(test_data[0]), np.array(test_data[1]), np.array(test_data[2])]

    print('============Load Dateset Success===========')
    return feat_col, train, val, test

