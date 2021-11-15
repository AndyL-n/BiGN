"""
author: L
date: 2021/11/9 15:28
"""
from SimGCN.dataloader import Loader
from SimGCN.model import SimGCN
import torch
import numpy as np
from tqdm import tqdm
from time import time, strftime, localtime
import pandas as pd
import os

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 64)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def Sampling(user, pos, dataset, neg_num):
    n_item = dataset.n_item
    all_pos = dataset.all_pos
    all_item = np.array(n_item)
    neg = []
    for u in user:
        p = np.ones(n_item)
        p[all_pos[u]] = 0
        p /= np.sum(p)
        neg_ = np.random.choice(all_item ,size = neg_num, p = p, replace=True)
        neg.append(neg_)

    neg = np.array(neg)
    neg = torch.from_numpy(neg)

    return neg

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
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
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_user(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # # 0/1 序列
    # r = getLabel(groundTrue, sorted_items)
    # ret = RecallPrecision_ATk(groundTrue, r, k)
    # return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

def test(model, dataset, params):
    topk = params['topk']
    batch_size = params['test_size']
    n_user = dataset.n_user
    testDict: dict = dataset.test_dict

    with torch.no_grad():
        users = list(testDict.keys())
        model.eval()
        try:
            assert batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = (len(users) - 1) // batch_size + 1
        for batch_users in minibatch(users, batch_size=batch_size):
            # train 数据
            allPos = dataset.get_user_pos(batch_users)
            # test 数据
            groundTrue = [testDict[u] for u in batch_users]

            batch_users = torch.Tensor(batch_users).long().to(params['device'])
            ratings = model.get_users_rating(batch_users).cpu()

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                # 行
                exclude_index.extend([range_i] * len(items))
                # 列
                exclude_items.extend(items)
            # train数据 改为特别小， 不参与预测
            ratings[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(ratings, k=topk)
            del ratings


            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)

        X = zip(rating_list, groundTrue_list)
        Recall, Precision, NDCG = 0, 0, 0

        for x in X:
            precision, recall, ndcg = test_one_user(x, topk)
            Recall += recall
            Precision += precision
            NDCG += ndcg

        Precision /= n_user
        Recall /= n_user
        NDCG /= n_user
        F1_score = 2 * (Precision * Recall) / (Precision + Recall)

        return F1_score, Precision, Recall, NDCG

if __name__ == "__main__":

    params = {}
    params['embed_size'] = 64
    params['name'] = 'SimGCN'
    params['dropout'] = 0
    params['neighbor_num'] = 10
    params['w'] = [1e-6, 1, 1e-6, 1]
    params['lr'] = 0.001
    params['batch_size'] = 1024
    GPU = torch.cuda.is_available()
    params['device'] = torch.device('cuda' if GPU else "cpu")
    params['epochs'] = 2000
    params['negative_num'] = 1500
    params['negative_weight'] = 900
    params['decay'] = 1e-4
    params['topk'] = 20
    params['test_size'] = 1024
    dataset = Loader(path="../Data/gowalla")
    model = SimGCN(params, dataset)
    model = model.to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    total_batch = (dataset.n_train - 1) // params['batch_size'] + 1
    print(f"Train on {dataset.n_train} samples,  {total_batch} steps per epoch")

    results = []
    # model.load_state_dict(t.load('weight/.tar'))
    result = test(dataset, model, params)

    F1, precision, recall, ndcg = test(model, dataset, params)
    print(F1, precision, recall, ndcg)
    results.append([0, 0, 0, 0, recall, ndcg, precision])
    timestamp = strftime('%Y-%m-%d', localtime(time()))
    path = '{}'.format(timestamp)
    print(path)

    for epoch in range(params['epochs']):
        t1 = time()
        # data = Sampling(dataset, params['negative_num'])
        user = torch.Tensor(dataset.train_user).long()
        pos = torch.Tensor(dataset.train_item).long()
        user, pos = shuffle(user, pos)

        aver_loss = 0
        for (batch_i, (batch_user, batch_pos)) in enumerate(
                minibatch(user, pos, batch_size=params['batch_size'])):
            batch_neg = Sampling(batch_user, batch_pos, dataset, params['negative_num'])
            model.zero_grad()

            loss, reg_loss = model.bpr_loss(batch_user, batch_pos, batch_neg)

            loss = loss + reg_loss * params['decay']
            loss.backward()
            optimizer.step()
            aver_loss += loss.cpu().item()
        aver_loss /= total_batch

        t2 = time()
        F1, precision, recall, ndcg = test(model, dataset, params)
        print(F1, precision, recall, ndcg)

        results.append([epoch + 1, t2 - t1, aver_loss, time() - t2, recall, ndcg, precision])
        pd.DataFrame(results,
                     columns=['Iteration', 'fit_time', 'loss', 'evaluate_time', 'recall', 'ndcg', 'precision']).to_csv(
            'log/' + path + '.csv')
        model.save_model('weight/' + path + '_epoch{}.tar'.format(epoch + 1))
        if epoch > 0:
            os.remove('weight/' + path + '_epoch{}.tar'.format(epoch))
