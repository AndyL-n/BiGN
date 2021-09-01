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

def load_data(file, embed_dim=8, test_neg_num=100):
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
    data_dict, test_dict = dict(), dict()
    exist_items = []
    with open(train_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                if len(line) > 1 and line[1] != '' :
                    items = [int(i) for i in line[1:]]
                # else:
                #     print(line)
                item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                data_dict[u_id] = items
                n_train += len(items)
                exist_items += items
        f.close()

    with open(test_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                if len(line) > 1 and line[1] != '' :
                    items = [int(i) for i in line[1:]]
                # else:
                #     print(line)
                item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                data_dict[u_id] = items + data_dict[u_id]
                n_test += len(items)
                exist_items += items
        f.close()

    user_num += 1
    item_num += 1
    feat_col = [sparseFeature('user_id', user_num, embed_dim), sparseFeature('item_id', item_num, embed_dim)]

    print('user_num: ' + str(user_num) + ',item_num：' + str(item_num) + ',interactions：' + str(n_train + n_test))

    print('============Negative Sampling===============')
    train_data, val_data, test_data = defaultdict(list), defaultdict(list), defaultdict(list)

    for u_id in tqdm(data_dict):
        pos_list = data_dict[u_id]
        neg_list = list(set(exist_items) - set(pos_list))
        if len(neg_list) >= len(pos_list) + test_neg_num:
            neg_list = random.sample(neg_list, len(pos_list) + test_neg_num)
        else:
            temp = [random.choice(neg_list) for _ in range(len(pos_list) + test_neg_num)]
            neg_list = temp
        val_idx, test_idx = random.randint(0,len(pos_list)), random.randint(0,len(pos_list))
        while val_idx == test_idx:
            test_idx = random.randint(0,len(pos_list))
        for index, item in enumerate(pos_list):
            if index == test_idx:
                test_data['user_id'].append(u_id)
                test_data['pos_id'].append(pos_list[index])
                test_data['neg_id'].append([neg_list[index]] + neg_list[len(pos_list):])
            elif index == val_idx:
                val_data['user_id'].append(u_id)
                val_data['pos_id'].append(pos_list[index])
                val_data['neg_id'].append(neg_list[index])
            else:
                train_data['user_id'].append(u_id)
                train_data['pos_id'].append(pos_list[index])
                train_data['neg_id'].append(neg_list[index])
        data_dict

        # shuffle 随机排序列表
    random.shuffle(train_data)
    random.shuffle(val_data)
    train = [np.array(train_data['user_id']), np.array(train_data['pos_id']), np.array(train_data['neg_id'])]
    val = [np.array(val_data['user_id']), np.array(val_data['pos_id']), np.array(val_data['neg_id'])]
    test = [np.array(test_data['user_id']), np.array(test_data['pos_id']), np.array(test_data['neg_id'])]
    print('============Data Preprocess End=============')

    return feat_col, train, val, test


