"""
author: L
date: 2021/8/31 11:47
"""

import torch as t
import numpy as np
import argparse
from temp.load_data_all import load_data
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from time import time, strftime, localtime
device_gpu = t.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument('--dataset', type=str, default='gowalla')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=20)
    # parser.add_argument('--act', type=str, default="leakyrelu")

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[8]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--reg', type=float, default=1e-4,
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    return parser.parse_args()

class LIGHT(nn.Module):
    def __init__(self, feature_columns, args, sparse_norm_adj, device):
        super(LIGHT, self).__init__()
        self.user_fea_col, self.item_fea_col = feature_columns
        self.n_user = self.user_fea_col['feat_num']
        self.n_item = self.item_fea_col['feat_num']
        self.device = device
        self.adj = sparse_norm_adj
        # self.node_dropout = args.node_dropout[0]
        # self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size
        # if args.act == "leakyrelu":
        #     self.act = nn.LeakyReLU(negative_slope=0.2)
        #     print("leakyrelu")
        # elif args.act == "relu":
        #     self.act = nn.ReLU()
        #     print("relu")
        self.layers = eval(args.layer_size)
        # self.decay = eval(args.regs)[0]
        self.decay = args.reg
        self.f = nn.Sigmoid()
        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.user_fea_col['embed_dim'])
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.item_fea_col['embed_dim'])
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        A_hat = self.adj

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        ego_embeddings = t.cat([users_emb, items_emb], 0)

        #torch.split(all_emb , [self.num_users, self.num_items])
        all_embeddings = [ego_embeddings]
        # if self.config['dropout']:
        #     if self.training:
        #         print("droping")
        #         g_droped = self.__dropout(self.keep_prob)
        #     else:
        #         g_droped = self.Graph
        # else:
        #     g_droped = self.Graph

        for k in range(len(self.layers)):
            # if self.A_split:
            #     temp_emb = []
            #     for f in range(len(g_droped)):
            #         temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
            #     side_emb = t.cat(temp_emb, dim=0)
            #     all_emb = side_emb
            # else:
            side_embeddings = t.sparse.mm(A_hat, ego_embeddings)

            all_embeddings += [side_embeddings]
        # all_embeddings = torch.cat(all_embeddings, 1)
        all_embeddings = t.stack(all_embeddings, 1)
        all_embeddings = t.mean(all_embeddings, dim=1, keepdim=False)
        users, items = t.split(all_embeddings, [self.n_user, self.n_item])
        return users, items

    def forward(self, user, item):
        # compute embedding
        all_user_emb, all_item_emb = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        user_emb = all_user_emb[user]
        item_emb = all_item_emb[item]
        inner_pro = t.mul(user_emb, item_emb)
        score = t.sum(inner_pro, dim=1)
        return score

    def get_embedding(self, users, pos, neg=None):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        users_emb_ego = self.embedding_user(users)
        pos_emb = all_items[pos]
        pos_emb_ego = self.embedding_item(pos)
        if neg == None:
            return users_emb, pos_emb, users_emb_ego, pos_emb_ego
        else:
            neg_emb = all_items[neg]
            neg_emb_ego = self.embedding_item(neg)
            return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def create_bpr_loss(self, user, pos, neg):
        (users_emb, pos_emb, neg_emb, user_emb_pre, pos_emb_pre, neg_emb_pre) = self.get_embedding(user.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (user_emb_pre.norm(2).pow(2) +
                              pos_emb_pre.norm(2).pow(2) +
                              neg_emb_pre.norm(2).pow(2)) / float(len(user))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = reg_loss*self.decay

        loss = loss + reg_loss

        return loss

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = t.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = t.from_numpy(sparse_mx.data)
    shape = t.Size(sparse_mx.shape)
    return t.sparse.FloatTensor(indices, values, shape)

class Data(Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.user = data[0]
        self.pos = data[1]
        self.neg = data[2]

    def __len__(self):
        return len(self.user)

    def __getitem__(self, item):
        user = self.user[item]
        pos = self.pos[item]
        neg = self.neg[item]
        return user, pos, neg

class Test(Dataset):
    def __init__(self, data):
        super(Test, self).__init__()
        self.user = data[0]
        self.pre = data[1]
        self.true = data[2]

    def __len__(self):
        return len(self.user)

    def __getitem__(self, item):
        user = self.user[item]
        pre = self.pre[item]
        true = self.true[item]
        return user, pre, true

def evaluate(model, dataset, test_size, k):
    hr, ndcg = 0, 0
    user_num = dataset.__len__()
    results = {'precision': np.zeros(1),
               'recall': np.zeros(1),
               'ndcg': np.zeros(1)}
    n_fold = user_num // test_size
    users_list = []
    rating_list = []
    groundTrue_list = []
    for flod in range(n_fold):
        batch_users, groundTrue, allPos = dataset[flod * test_size : (flod + 1) * test_size]
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

    # batch_users, groundTrue, allPos = dataset[(n_fold + 1) * test_size:]
    # batch_users_gpu = t.Tensor(batch_users).long()
    # rating = model.getUsersRating(batch_users_gpu)
    # exclude_index = []
    # exclude_items = []
    # for range_i, items in enumerate(allPos):
    #     exclude_index.extend([range_i] * len(items))
    #     exclude_items.extend(items)
    # #  无限小
    # rating[exclude_index, exclude_items] = -(1 << 10)
    # _, rating_K = t.topk(rating, k=k)
    # rating = rating.cpu().numpy()
    # del rating
    # users_list.append(batch_users)
    # rating_list.append(rating_K.cpu())
    # groundTrue_list.append(groundTrue)
    # X = zip(rating_list, groundTrue_list)
    # pre_results = []
    # for x in X:
    #     pre_results.append(test_one_batch(x))
    # for result in pre_results:
    #     results['recall'] += result['recall']
    #     results['precision'] += result['precision']
    #     results['ndcg'] += result['ndcg']

    results['recall'] /= float(user_num)
    results['precision'] /= float(user_num)
    results['ndcg'] /= float(user_num)


    return results['recall'], results['ndcg']

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    # print(groundTrue)
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
    np.random.seed(2020)
    t.manual_seed(2020)
    t.cuda.manual_seed(2020)
    args = parse_args()
    print(args)
    feature_columns, train, val, test, adj = load_data(args.dataset, args.embed_size)
    sparse_norm_adj = sparse_mx_to_torch_sparse_tensor(adj)
    model = LIGHT(feature_columns, args, sparse_norm_adj, device_gpu).to(device_gpu)
    # print(model)
    """
       *********************************************************
       Train.
       """

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = Data(train)
    val_dataset = Data(val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataset = Test(test)
    HR_best, NDCG_best = 0, 0
    results = []
    for epoch in range(args.epoch):
        epoch_loss, val_loss = 0, 0
        t1 = time()
        for user, pos, neg in train_loader:
            # user_embed, pos_embed, neg_embed = model(sparse_norm_adj, user.long(), pos.long(), neg.long(), drop_flag=True)
            loss = model.create_bpr_loss(user, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        for user, pos, neg in val_loader:
            loss = model.create_bpr_loss(user, pos, neg)
            val_loss += loss.item()
        t2 = time()
        print("epoch = %d, loss = %.4f, val_loss = %.4f"%(epoch+1, epoch_loss, val_loss))
        HR, NDCG = evaluate(model, test_dataset, args.test_size, args.top_k)
        # print(HR, NDCG)
        # HR, NDCG = 0,0
        t3 = time()
        print("HR = %.4f, NDCG = %.4f"%(HR, NDCG))
        results.append([epoch, t2 - t1, epoch_loss, val_loss, t3 - t2, HR, float(NDCG)])
        HR_best, NDCG_best = max(HR, HR_best), max(int(NDCG), NDCG_best)
        timestamp = strftime('%Y-%m-%d-%H-%M', localtime(time()))
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'epoch_loss', 'val_loss', 'evaluate_time', 'hit_rate', 'ndcg']) \
        .to_csv('log/{}_log_{}_dim_{}_K_{}_{}.csv' \
        .format('LightGCN', args.dataset, args.embed_size, args.top_k, timestamp), index=False)
