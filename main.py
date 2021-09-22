"""
author: L
date: 2021/9/6 17:07
"""
from register import dataset, args, MODELS
from utils import set_seed, minibatch, RecallPrecision_ATk, getLabel, NDCGatK_r, sample, shuffle
import numpy as np
import torch as t
import pandas as pd
from time import time, strftime, localtime
import torch.optim as optim

def test_one_user(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # 0/1 序列
    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg = [], [], []
    for k in eval(args.topks):
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(dataset, model):
    batch_size = args.test_batch
    testDict: dict = dataset.testDict
    model: model.LightGCN
    # eval mode with no dropout
    model = model.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with t.no_grad():
        users = list(testDict.keys())
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
            allPos = dataset.getUserPosItems(batch_users)
            # test 数据
            groundTrue = [testDict[u] for u in batch_users]
            # batch内的user [batch * 1]
            batch_users_gpu = t.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)
            # batch所有的评分[batch * n_items]
            rating = model.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                # 行
                exclude_index.extend([range_i] * len(items))
                # 列
                exclude_items.extend(items)
            # train数据 改为特别小， 不参与预测
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = t.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_user(x))
        scale = float(batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        return results

if __name__ == '__main__':
    print(">>SEED:", args.seed)
    set_seed(args.seed)
    model = MODELS[args.model_name](args, dataset)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    results = []
    result = Test(dataset, model)
    precision, recall, ndcg = [result[x] for x in result]
    print(precision, recall, ndcg)
    results.append([0, 0, 0, 0, recall, ndcg, precision])
    timestamp = strftime('%Y-%m-%d', localtime(time()))

    print('start training...')

    total_batch = (dataset.traindataSize - 1) // args.train_batch + 1
    print(f"Train on {dataset.traindataSize} samples,  {total_batch} steps per epoch")

    for epoch in range(args.epochs):
        t1 = time()
        S = sample(dataset)
        users = t.Tensor(S[:, 0]).long()
        posItems = t.Tensor(S[:, 1]).long()
        negItems = t.Tensor(S[:, 2]).long()

        users = users.to(args.device)
        posItems = posItems.to(args.device)
        negItems = negItems.to(args.device)
        users, posItems, negItems = shuffle(users, posItems, negItems)

        aver_loss = 0.
        for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                minibatch(users, posItems, negItems, batch_size=args.train_batch)):
            loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
            reg_loss = reg_loss * args.decay
            loss = loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(batch_i, loss)
            aver_loss += loss.cpu().item()
        aver_loss = aver_loss / total_batch
        print(f'EPOCH[{epoch + 1}/{args.epochs}] loss{aver_loss:.10f}')

        t2 = time()
        result = Test(dataset, model)
        precision, recall, ndcg = [result[x] for x in result]
        print(precision, recall, ndcg)
        results.append([epoch + 1, t2-t1, aver_loss, time()-t2, recall, ndcg, precision])
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'loss', 'evaluate_time', 'recall', 'ndcg', 'precision'])\
            .to_csv('log/{}_{}_layer{}_dim{}_batch{}_K{}_lr{}_{}.csv'
                    .format(args.model_name, args.dataset, args.layer, args.embed_size, args.train_batch, args.topks, args.lr, timestamp), index=False)
