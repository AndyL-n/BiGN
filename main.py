"""
author: L
date: 2021/9/6 17:07
"""
from register import MODELS
from dataloader import Loader


from utils import set_seed, minibatch, RecallPrecision_ATk, getLabel, NDCGatK_r, sample, shuffle
import numpy as np
import torch as t
import pandas as pd
from time import time, strftime, localtime
import torch.optim as optim
import sys
import os
import argparse
import configparser

def test_one_user(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # 0/1 序列
    r = getLabel(groundTrue, sorted_items)

    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(dataset, model, params):
    batch_size = params['test_batch_size']
    testDict: dict = dataset.test_dict
    model: model.LightGCN
    # eval mode with no dropout
    if params['name'] == 'GF_CF' or params['name'] == 'LGCN_IDE':
        model.train()
    else:
        model = model.eval()

    topks = [params['topk']]
    # topks = eval(topks)
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
            allPos = dataset.get_user_pos(batch_users)
            # test 数据
            groundTrue = [testDict[u] for u in batch_users]
            # batch内的user [batch * 1]
            batch_users_gpu = t.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(params['device'])
            # batch所有的评分[batch * n_items]
            rating = model.get_users_rating(batch_users_gpu)
            # print(rating.shape)
            # print(rating)
            # sys.exit(
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
            pre_results.append(test_one_user(x, topks))
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

def data_param_prepare(model, dataset):
    config = configparser.ConfigParser()
    config.read('params/' + model + '_' + dataset + '_config.ini')
    print('params/' + model + '_' + dataset + '_config.ini')
    all_dataset = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
    params = {}
    params['name'] = config['Model']['name']
    params['dataset'] = config['Training']['dataset']
    GPU = t.cuda.is_available()
    device = t.device('cuda' if GPU else "cpu")
    params['device'] = device
    # if model == 'GC_CF':


    params['embed_size'] = config.getint('Model', 'embedding_dim')
    # params['layer'] = config.getint('Model', 'layers')
    params['test_batch_size'] = config.getint('Testing', 'test_batch_size')
    params['topk'] = config.getint('Testing', 'topk')
    # params['lr'] = config.getfloat('Training', 'lea2rning_rate')
    # params['seed'] = config.getint('Model', 'seed')
    # params['dropout'] = config.getint('Model', 'dropout')


    if params['dataset'] in all_dataset:
        dataset = Loader(path="Data/" + params['dataset'])
    else:
        sys.exit("No such file or directory:" + params['dataset'])
    return params, dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GF_CF', type=str)
    parser.add_argument('--dataset', default='gowalla', type=str)
    args = parser.parse_args()
    params, dataset = data_param_prepare(args.model, args.dataset)
    model = params['name']
    print('================Run\t' + model + '\t🏃================')

    if model == 'GF_CF' or model == 'LGCN_IDE':
        model = MODELS[model](params, dataset)
        result = Test(dataset, model, params)
        precision, recall, ndcg = [result[x] for x in result]
        print(precision, recall, ndcg)
    else:
        print(">>SEED:", params['seed'])
        set_seed(params['seed'])

        model = MODELS[model](params, dataset)
        model = model.to(params['device'])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])


        results = []
        # model.load_state_dict(t.load('weight/.tar'))
        result = Test(dataset, model, params)

        precision, recall, ndcg = [result[x] for x in result]
        print(recall, ndcg, precision)
        exit()
        results.append([0, 0, 0, 0, recall, ndcg, precision])
        timestamp = strftime('%Y-%m-%d', localtime(time()))
        path = '{}_{}_layer{}_dim{}_batch{}_K{}_lr{}_neighbor{}_{}' \
            .format(args.model_name, args.dataset, args.layer, args.embed_size, args.train_batch, args.topks, args.lr, args.neighbor, timestamp)
        print(path)
        print('start training...')

        total_batch = (dataset.n_train - 1) // args.train_batch + 1
        print(f"Train on {dataset.n_train} samples,  {total_batch} steps per epoch")


        for epoch in range(args.epochs):
            t1 = time()
            S = sample(dataset)
            users = t.Tensor(S[:, 0]).long()
            posItems = t.Tensor(S[:, 1]).long()
            negItems = t.Tensor(S[:, 2:]).long()
            users = users.to(args.device)
            posItems = posItems.to(args.device)
            negItems = negItems.to(args.device)
            users, posItems, negItems = shuffle(users, posItems, negItems)

            aver_loss = 0.
            for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
                    minibatch(users, posItems, negItems, batch_size=args.train_batch)):
                loss, reg_loss = model.bpr_loss(batch_users, batch_pos, batch_neg)
                # loss,reg_loss = model.css_loss(batch_users, batch_pos, batch_neg)
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

            # exit()
            print(recall, ndcg, precision)
            results.append([epoch + 1, t2-t1, aver_loss, time()-t2, recall, ndcg, precision])
            pd.DataFrame(results, columns=['Iteration', 'fit_time', 'loss', 'evaluate_time', 'recall', 'ndcg', 'precision']).to_csv('log/'+ path +'.csv')
            model.save_model('weight/' + path + '_epoch{}.tar'.format(epoch + 1))
            if epoch > 0:
                os.remove('weight/' + path + '_epoch{}.tar'.format(epoch))