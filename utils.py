"""
author: L
date: 2021/9/6 9:10
"""

import numpy as np
import torch as t
from LightGCN_torch import LIGHT, sparse_mx_to_torch_sparse_tensor, parse_args, Test
from load_data_all import load_data
device_gpu = t.device("cpu")

def evaluate(model, dataset, test_size, k):
    user_num = dataset.__len__()
    results = {'precision': np.zeros(1),
               'recall': np.zeros(1),
               'ndcg': np.zeros(1)}
    users_list = []
    rating_list = []
    groundTrue_list = []
    left, right = 0, test_size
    while True:
        batch_users, groundTrue, allPos = dataset[left: right]
        batch_users_gpu = t.Tensor(batch_users).long()
        rating = model.getUsersRating(batch_users_gpu)
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        #  无限小
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = t.topk(rating, k=k)
        rating = rating.detach().numpy()
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, k))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']

        break

        right += test_size
        left += test_size
        if (right > user_num):
            break
    print(users_list[0])
    print(len(users_list))
    results['recall'] /= float(user_num)
    results['precision'] /= float(user_num)
    results['ndcg'] /= float(user_num)


    return results['recall'], results['ndcg']

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        print(groundTrue)
        predictTopK = pred_data[i]
        print(predictTopK)
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        print(pred)
        pred = np.array(pred).astype("float")
        r.append(pred)
    print(r[0])
    return np.array(r).astype('float')

def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    print()
    print(len(groundTrue))
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    ret = RecallPrecision_ATk(groundTrue, r, k)

    pre.append(ret['precision'])
    recall.append(ret['recall'])
    ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    print(right_pred)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

if __name__ == '__main__':
    np.random.seed(29)
    t.manual_seed(29)
    t.cuda.manual_seed(29)
    args = parse_args()
    args.dataset = 'ml-100k'
    args.test_size = 64
    print(args)
    feature_columns, train, val, test, adj = load_data(args.dataset, args.embed_size)
    sparse_norm_adj = sparse_mx_to_torch_sparse_tensor(adj)
    model = LIGHT(feature_columns, args, sparse_norm_adj, device_gpu).to(device_gpu)
    test_dataset = Test(test)
    HR, NDCG = evaluate(model, test_dataset, args.test_size, args.top_k)
    print(HR,NDCG)