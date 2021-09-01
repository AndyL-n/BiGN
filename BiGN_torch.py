"""
author: L
date: 2021/9/1 14:43
"""

import torch as t
import numpy as np
import argparse
from load_data_loo import load_data
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from time import time, strftime, localtime

device_gpu = t.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BiGN")

    parser.add_argument('--dataset', type=str, default='ml-100k')

    parser.add_argument('--top_k', type=int, default=10)
    # parser.add_argument('--act', type=str, default="leakyrelu")

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[8]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--test_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--reg', type=float, default=0.001,
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    return parser.parse_args()

class BiGN(nn.Module):
    def __init__(self, feature_columns, args, device):
        super(BiGN, self).__init__()
        self.user_fea_col, self.item_fea_col = feature_columns
        self.n_user = self.user_fea_col['feat_num']
        self.n_item = self.item_fea_col['feat_num']

        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        # self.node_dropout = args.node_dropout[0]
        # self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size
        self.layers = eval(args.layer_size)
        self.decay = args.reg

        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(self.n_user, self.emb_size))),
            'item_emb': nn.Parameter(initializer(t.empty(self.n_item, self.emb_size)))
        })

    def forward(self, sparse_user_adj, sparse_item_adj, users, pos_items, neg_items):

        user_hat = sparse_user_adj
        item_hat = sparse_item_adj

        user_ego_embeddings = self.embedding_dict['user_emb']
        item_ego_embeddings = self.embedding_dict['item_emb']

        user_all_embeddings = [user_ego_embeddings]
        item_all_embeddings = [item_ego_embeddings]

        for k in range(len(self.layers)):
            user_side_embeddings = t.sparse.mm(user_hat, user_ego_embeddings)
            item_side_embeddings = t.sparse.mm(item_hat, item_ego_embeddings)

            user_all_embeddings += [user_side_embeddings]
            item_all_embeddings += [item_side_embeddings]

        # all_embeddings = torch.cat(all_embeddings, 1)
        user_all_embeddings = t.stack(user_all_embeddings, 1)
        u_g_embeddings = t.mean(user_all_embeddings, dim=1, keepdim=False)

        item_all_embeddings = t.stack(item_all_embeddings, 1)
        i_g_embeddings = t.mean(item_all_embeddings, dim=1, keepdim=False)


        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def getEmbeds(self, sparse_user_adj, sparse_item_adj):

        user_hat = sparse_user_adj
        item_hat = sparse_item_adj

        user_ego_embeddings = self.embedding_dict['user_emb']
        item_ego_embeddings = self.embedding_dict['item_emb']

        user_all_embeddings = [user_ego_embeddings]
        item_all_embeddings = [item_ego_embeddings]

        for k in range(len(self.layers)):
            user_side_embeddings = t.sparse.mm(user_hat, user_ego_embeddings)
            item_side_embeddings = t.sparse.mm(item_hat, item_ego_embeddings)

            user_all_embeddings += [user_side_embeddings]
            item_all_embeddings += [item_side_embeddings]

        # all_embeddings = torch.cat(all_embeddings, 1)
        user_all_embeddings = t.stack(user_all_embeddings, 1)
        u_g_embeddings = t.mean(user_all_embeddings, dim=1, keepdim=False)

        item_all_embeddings = t.stack(item_all_embeddings, 1)
        i_g_embeddings = t.mean(item_all_embeddings, dim=1, keepdim=False)

        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, user, pos, neg):
        pos_scores = t.sum(t.mul(user, pos), axis=1)
        neg_scores = t.sum(t.mul(user, neg), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * t.mean(maxi)

        # cul regularizer
        regularizer = (t.norm(user) ** 2 + t.norm(pos) ** 2 + t.norm(neg) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss#, emb_loss

    def getScores(self, users, pos_items, neg_items):
        pos_scores = t.mul(users, pos_items)
        neg_scores = t.mul(users, neg_items)
        pos_scores = t.sum(pos_scores, axis=2)
        neg_scores = t.sum(neg_scores, axis=2)
        scores = t.cat([pos_scores, neg_scores], 1)
        return scores

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
        self.pos = data[1]
        self.neg = data[2]

    def __len__(self):
        return len(self.user)

    def __getitem__(self, item):
        user = self.user[item]
        pos = self.pos[item]
        neg = self.neg[item]
        return user, pos, neg

def evaluate(model, sparse_user_adj, sparse_item_adj, loader, k, feature_columns):
    hr, ndcg = 0, 0
    user_num = feature_columns[0]['feat_num']
    item_num = feature_columns[1]['feat_num']
    t = 1
    for user, pos, neg in loader:
        u_e, i_e = model.getEmbeds(sparse_user_adj, sparse_item_adj)
        user = user.long()
        pos = pos.long()
        neg = neg.long()
        user_embed = u_e[user]
        pos_embed = i_e[pos]
        neg_embed = i_e[neg]
        user_embed = user_embed.unsqueeze(1)
        pos_embed = pos_embed.unsqueeze(1)
        scores = - model.getScores(user_embed, pos_embed, neg_embed)
        rank = scores.argsort()[:,0].cpu().numpy()
        rank = list(rank)
        for r in rank:
            if r < k:
                hr += 1
                ndcg += 1 / np.log2(r + 2)

    return hr/user_num, ndcg/user_num

if __name__ == '__main__':
    np.random.seed(29)
    t.manual_seed(29)
    t.cuda.manual_seed(29)
    args = parse_args()
    print(args)
    feature_columns, train, val, test, user_sim, item_sim = load_data(args.dataset, args.embed_size)
    model = BiGN(feature_columns, args, device_gpu).to(device_gpu)

    # print(model)
    """
       *********************************************************
       Train.
       """
    cur_best_pre_0, stopping_step = 0, 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sparse_user_adj = sparse_mx_to_torch_sparse_tensor(user_sim)
    sparse_item_adj = sparse_mx_to_torch_sparse_tensor(item_sim)
    # print(sparse_norm_adj)
    train_dataset = Data(train)
    val_dataset = Data(val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataset = Test(test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers=0)
    HR_best, NDCG_best = 0, 0
    results = []
    for epoch in range(args.epoch):
        epoch_loss = 0
        t1 = time()
        for user, pos, neg in train_loader:
            user_embed, pos_embed, neg_embed = model(sparse_user_adj, sparse_item_adj, user.long(), pos.long(), neg.long())
            loss, mf_loss = model.create_bpr_loss(user_embed, pos_embed, neg_embed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += mf_loss.item()
        for user, pos, neg in val_loader:
            user_embed, pos_embed, neg_embed = model(sparse_user_adj, sparse_item_adj, user.long(), pos.long(), neg.long())
            loss, mf_loss = model.create_bpr_loss(user_embed, pos_embed, neg_embed)
            val_loss = loss
        t2 = time()
        print("epoch = %d, loss = %.4f, val_loss = %.4f" % (epoch + 1, epoch_loss, val_loss))
        HR, NDCG = evaluate(model, sparse_user_adj, sparse_item_adj, test_loader, args.top_k, feature_columns)
        t3 = time()
        print("HR = %.4f, NDCG = %.4f" % (HR, NDCG))
        results.append([epoch, t2 - t1, epoch_loss, val_loss, t3 - t2, HR, float(NDCG)])
        HR_best, NDCG_best = max(HR, HR_best), max(int(NDCG), NDCG_best)
        timestamp = strftime('%Y-%m-%d-%H-%M', localtime(time()))
    pd.DataFrame(results,
                 columns=['Iteration', 'fit_time', 'epoch_loss', 'val_loss', 'evaluate_time', 'hit_rate', 'ndcg']) \
        .to_csv('log/{}_log_{}_dim_{}_K_{}_{}.csv' \
                .format('BiGN', args.dataset, args.embed_size, args.top_k, timestamp), index=False)