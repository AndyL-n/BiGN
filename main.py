"""
author: L
date: 2021/9/6 17:07
"""
from register import dataset, args, MODELS
from utils import set_seed, BPRLoss, minibatch, RecallPrecision_ATk, getLabel, NDCGatK_r, timer, UniformSample_original, shuffle
import numpy as np
import torch as t
import pandas as pd
from time import time, strftime, localtime


def Train(dataset, recommend_model, loss_class, neg_k=1, ):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class

    with timer(name="Sample"):
        S = UniformSample_original(dataset)
    users = t.Tensor(S[:, 0]).long()
    posItems = t.Tensor(S[:, 1]).long()
    negItems = t.Tensor(S[:, 2]).long()

    users = users.to(args.device)
    posItems = posItems.to(args.device)
    negItems = negItems.to(args.device)
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch = len(users) // args.train_batch + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            minibatch(users, posItems, negItems, batch_size=args.train_batch)):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        print(batch_i, cri)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.10f}-{time_info}" , aver_loss

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
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

def Test(dataset, Recmodel):
    u_batch_size = args.test_batch
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with t.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = t.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
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
            pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
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
    bpr = BPRLoss(model, args)
    Neg_k = 1
    results = []
    result = Test(dataset, model)
    precision, recall, ndcg = [result[x] for x in result]
    print(precision, recall, ndcg)
    results.append([0, 0, 0, 0, recall, ndcg, precision])
    timestamp = strftime('%Y-%m-%d', localtime(time()))

    print('start training...')
    for epoch in range(args.epochs):
        t1 = time()
        output, loss = Train(dataset, model, bpr, neg_k=Neg_k)
        print(loss)
        print(f'EPOCH[{epoch + 1}/{args.epochs}] {output}')
        t2 = time()
        result = Test(dataset, model)
        precision, recall, ndcg = [result[x] for x in result]
        print(precision, recall, ndcg)
        results.append([epoch + 1, t2-t1, loss, time()-t2, recall, ndcg, precision])
        pd.DataFrame(results, columns=['Iteration', 'fit_time', 'loss', 'evaluate_time', 'recall', 'ndcg', 'precision'])\
            .to_csv('log/{}_{}_layer{}_dim{}_K{}_lr{}_{}.csv'
                    .format(args.model_name, args.dataset, args.layer, args.embed_size, args.topks, args.lr, timestamp), index=False)
