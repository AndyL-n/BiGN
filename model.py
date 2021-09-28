import torch as torch
import torch as t
import torch.nn.functional as F
from torch import nn
from parse import cprint
from sys import exit
from sparsesvd import sparsesvd
import time
import numpy as np
import scipy.sparse as sp

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_users_rating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class LightGCN(BasicModel):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.split = self.args.split
        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)
        
        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.args.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.layer):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
            else:
                side_emb = t.sparse.mm(g_droped, all_emb)

            all_emb = side_emb + all_emb if self.args.residual else side_emb
            embs.append(all_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class NGCF(BasicModel):
    def __init__(self, args, dataset):
        super(NGCF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.layer_size = eval(self.args.layer_size)
        self.mess_dropout = eval(self.args.mess_dropout)
        self.split = self.args.split
        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        # xavier init
        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            cprint('use xavier initilizer')

        # weight
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.embed_size] + self.layer_size
        for k in range(self.layer):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(t.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(t.empty(1, layers[k + 1])))})

            weight_dict.update({'W_bi_%d' % k: nn.Parameter(initializer(t.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_bi_%d' % k: nn.Parameter(initializer(t.empty(1, layers[k + 1])))})

        self.weight_dict = weight_dict


        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for NGCF
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]

        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.args.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.layer):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
            else:
                side_emb = t.sparse.mm(g_droped, all_emb)

            # transformed sum messages of neighbors.
            sum_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % layer]) + self.weight_dict['b_gc_%d' % layer]

            # bi messages of neighbors.
            # element-wise product
            bi_emb = torch.mul(all_emb, side_emb)

            # transformed bi messages of neighbors.
            bi_emb = torch.matmul(bi_emb, self.weight_dict['W_bi_%d' % layer]) + self.weight_dict['b_bi_%d' % layer]

            # non-linear activation.
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb + bi_emb)

            # message dropout.
            all_emb = nn.Dropout(self.mess_dropout[layer])(all_emb)

            # normalize the distribution of embeddings.
            norm_emb = F.normalize(all_emb, p=2, dim=1)

            # all_emb = side_emb + all_emb if self.args.residual else side_emb
            embs.append(norm_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class BiGN(BasicModel):
    def __init__(self, args, dataset):
        super(BiGN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.keep_prob = self.args.keep_prob
        self.split = self.args.split
        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = t.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class DGCN_HN(BasicModel):
    def __init__(self, args, dataset):
        super(DGCN_HN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keep_prob
        self.A_split = self.args.split
        self.embedding_user = t.nn.Embedding(
            num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(
            num_embeddings=self.n_item, embedding_dim=self.embed_size)
        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.LGraph = self.dataset.getSparseLGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
            l_graph = []
            for g in self.LGraph:
                l_graph.append(self.__dropout(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
            l_graph = self.__dropout_x(self.LGraph, keep_prob)
        return graph, l_graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        print(all_emb.shape)
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
                l_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
                l_droped = self.LGraph
        else:
            g_droped = self.Graph
            l_droped = self.LGraph
            print("-----------------")

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb, temp_L_emb = [], []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                    temp_L_emb.append(t.sparse.mm(l_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
                side_L_emb = t.cat(temp_L_emb, dim=0)

            else:
                side_emb = t.sparse.mm(g_droped, all_emb)
                side_L_emb = t.sparse.mm(l_droped, all_emb)
            attention_s = t.mean((side_emb * all_emb + side_emb) , dim=1)
            attention_l = t.mean((side_L_emb * all_emb + side_L_emb) , dim=1)
            attention = t.exp(attention_s) + t.exp(attention_l)
            attention_s = t.exp(attention_s) / attention
            attention_l = t.exp(attention_l) / attention
            attention_s = attention_s.unsqueeze(1)
            attention_l = attention_l.unsqueeze(1)
            print(attention_s)
            print(side_emb)
            print(attention_s*side_emb)
            print(attention_s+attention_l)
            all_emb = all_emb + (attention_s * side_emb) + (attention_l * side_L_emb)
            print(all_emb.shape)
            exit()
            embs.append(all_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class GCN(BasicModel):
    def __init__(self, args, dataset):
        super(GCN, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.layer_size = eval(self.args.layer_size)
        self.mess_dropout = eval(self.args.mess_dropout)
        self.split = self.args.split

        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        # xavier init
        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            cprint('use xavier initilizer')

        # weight
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.embed_size] + self.layer_size
        for k in range(self.layer):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(t.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(t.empty(1, layers[k + 1])))})

        self.weight_dict = weight_dict

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

    def computer(self):
        """
        propagate methods for GCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]

        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.args.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.layer):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
            else:
                side_emb = t.sparse.mm(g_droped, all_emb)

            # transformed sum messages of neighbors.
            sum_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % layer]) + self.weight_dict['b_gc_%d' % layer]

            # non-linear activation.
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb)

            # message dropout.
            all_emb = nn.Dropout(self.mess_dropout[layer])(all_emb)

            embs.append(all_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class GCMC(BasicModel):
    def __init__(self, args, dataset):
        super(GCMC, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.layer_size = eval(self.args.layer_size)
        self.mess_dropout = eval(self.args.mess_dropout)
        self.split = self.args.split

        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        # xavier init
        if self.args.pretrain:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        else:
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            cprint('use xavier initilizer')

        # weight
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict()
        layers = [self.embed_size] + self.layer_size
        for k in range(self.layer):
            weight_dict.update({'W_gc_%d' % k: nn.Parameter(initializer(t.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_gc_%d' % k: nn.Parameter(initializer(t.empty(1, layers[k + 1])))})

            weight_dict.update({'W_mlp_%d' % k: nn.Parameter(initializer(t.empty(layers[k], layers[k + 1])))})
            weight_dict.update({'b_mlp_%d' % k: nn.Parameter(initializer(t.empty(1, layers[k + 1])))})

        self.weight_dict = weight_dict

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

    def computer(self):
        """
        propagate methods for GCMC
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        #   t.split(all_emb , [self.n_user, self.n_item])
        embs = [all_emb]

        if self.args.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.args.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.layer):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
            else:
                side_emb = t.sparse.mm(g_droped, all_emb)

            # transformed sum messages of neighbors.
            sum_emb = torch.matmul(side_emb, self.weight_dict['W_gc_%d' % layer]) + self.weight_dict['b_gc_%d' % layer]

            # non-linear activation.
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb)

            # message dropout.
            all_emb = nn.Dropout(self.mess_dropout[layer])(all_emb)

            # dense layer.
            mlp_emb = torch.matmul(all_emb, self.weight_dict['W_mlp_%d' % layer]) + self.weight_dict['b_mlp_%d' % layer]

            embs.append(mlp_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.n_user, self.n_item])
        return users, items

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = t.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = t.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def get_embedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = t.mul(users_emb, pos_emb)
        pos_scores = t.sum(pos_scores, dim=1)
        neg_scores = t.mul(users_emb, neg_emb)
        neg_scores = t.sum(neg_scores, dim=1)

        loss = t.mean(t.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        scores = t.mul(users_emb, items_emb)
        scores = t.sum(scores, dim=1)
        return scores

class Two_tower(BasicModel):
    def __init__(self, args, dataset):
        super(Two_tower, self).__init__()
        self.args = args
        self.dataset = dataset

class BPRMF(BasicModel):
    def __init__(self, args, dataset):
        super(BPRMF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size

        self.embedding_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)
        print("using Normal distribution N(0,1) initialization for BPRMF")

        self.f = nn.Sigmoid()
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

    def get_users_rating(self, users):
        users = users.long()
        # [batch_szie * embed_size]
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        # [batch_szie * embed_size][embed_size * n_item] = [batch_szie * n_item]
        scores = t.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())       # [batch_szie * embed_size]
        pos_emb = self.embedding_item(pos.long())           # [batch_szie * embed_size]
        neg_emb = self.embedding_item(neg.long())           # [batch_szie * embed_size]
        pos_scores = t.sum(t.mul(users_emb, pos_emb), dim=-1, keepdim=True)
        neg_scores = t.sum(t.mul(users_emb, neg_emb), dim=-1, keepdim=True)
        # loss = t.mean(-1.0 * nn.functional.LogSigmoid(neg_scores - pos_scores))
        loss = t.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LGCN_IDE(BasicModel):
    def __init__(self, args, dataset):
        super(LGCN_IDE, self).__init__()
        self.adj_mat = dataset.R.tolil()
        self.args = args

    def train(self):
        adj_mat = self.adj_mat
        # print(adj_mat.shape)  # adj           [n_user, n_item]
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))  # Du
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # Du^-0.5       [n_user, n_user]
        d_mat_i = d_mat
        norm_adj = d_mat.dot(adj_mat)  # Du^-0.5 * R

        colsum = np.array(adj_mat.sum(axis=0))  # Di
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # Di^-0.5       [n_item, n_item]
        d_mat_u = d_mat
        d_mat_u_inv = sp.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)  # Du^-0.5 * R * Di^-0.5
        self.norm_adj = norm_adj.tocsr()  # [n_user * n_item]
        end = time.time()
        print('training time for LGCN-IDE', end - start)

    def get_users_rating(self, batch_users):
        norm_adj = self.norm_adj
        batch_test = np.array(norm_adj[batch_users, :].todense())
        U_1 = batch_test @ norm_adj.T @ norm_adj  # [batch, n_item] * [n_item, n_user] * [n_user, n_item] = [batch, n_item]
        if (self.args.dataset == 'gowalla'):
            U_2 = U_1 @ norm_adj.T @ norm_adj  # [batch, n_item] * [n_item, n_user] * [n_user, n_item] = [batch, n_item]
            return torch.from_numpy(U_2)
        else:
            return torch.from_numpy(U_1)

class GF_CF(BasicModel):
    def __init__(self, args, dataset):
        super(GF_CF, self).__init__()
        self.adj_mat = dataset.R.tolil()
        self.args = args

    def train(self):
        print("train GF_CF...")
        adj_mat = self.adj_mat  # adj           [n_user, n_item]
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))  # Du
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # Du^-0.5       [n_user, n_user]
        norm_adj = d_mat.dot(adj_mat)  # Du ^ -0.5 * R

        colsum = np.array(adj_mat.sum(axis=0))  # Di
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat  # Di^-0.5       [n_item, n_item]
        self.d_mat_i_inv = sp.diags(1 / d_inv)  # Di^0.5        [n_item, n_item]
        norm_adj = norm_adj.dot(d_mat)  # Du ^ -0.5 * R * Di ^ -0.5
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)  # 奇异值分解
        # [k, n_user] * [k, k] * [k, n_item]
        end = time.time()
        print('training time for GF-CF', end - start)

    def get_users_rating(self, batch_users):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users, :].todense())
        U_2 = batch_test @ norm_adj.T @ norm_adj
        if (self.args.dataset == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            # [batch, n_item] * [n_item, n_item] * [n_item, k] * [k, n_item] * [n_item, n_item] = [batch, n_item]
            ret = U_2 + 0.3 * U_1

        return torch.from_numpy(ret)

class NeuMF(BasicModel):
    def __init__(self, args, dataset):
        super(NeuMF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        # self.layer = self.args.layer
        self.layer = 1
        self.embedding_GMF_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_GMF_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        self.embedding_MLP_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_MLP_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        nn.init.normal_(self.embedding_GMF_user.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_GMF_item.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_MLP_user.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_MLP_item.weight, mean=0., std= 0.01)
        cprint('use normal initilizer')

        # Layer configuration
        ##  MLP Layers
        MLP_layers = []
        layers_shape = [self.embed_size * 2]
        for i in range(self.layer):
            layers_shape.append(layers_shape[-1] // 2)
            MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
            MLP_layers.append(nn.ReLU())

        print("MLP Layer Shape ::", layers_shape)
        self.MLP_layers = nn.Sequential(*MLP_layers)

        ## Final Layer
        self.final_layer = nn.Linear(layers_shape[-1] * 2, 1)

        # Layer initialization
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.final_layer.weight, a=1, nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})")

    def get_users_rating(self, users):
        batch_size = users.shape[0]
        user_GMF_emb = self.embedding_GMF_user(users.long())
        user_MLP_emb = self.embedding_MLP_user(users.long())
        item_GMF_emb = self.embedding_GMF_item.weight
        item_MLP_emb = self.embedding_MLP_item.weight
        user_GMF_emb = torch.repeat_interleave(user_GMF_emb.unsqueeze(dim=1), repeats=self.n_item, dim=1)
        item_GMF_emb = item_GMF_emb.unsqueeze(0)
        user_MLP_emb = torch.repeat_interleave(user_MLP_emb.unsqueeze(dim=1), repeats=self.n_item, dim=1)
        item_MLP_emb = torch.repeat_interleave(item_MLP_emb.unsqueeze(dim=0), repeats=batch_size, dim=0)
        GMF_vector = torch.mul(user_GMF_emb, item_GMF_emb)  # [batch_size, n_item, dim]

        MLP_vector = torch.cat([user_MLP_emb, item_MLP_emb],dim=2) # [batch_size, n_item, 2*dim]
        MLP_vector = self.MLP_layers(MLP_vector)

        vector = torch.cat([GMF_vector, MLP_vector], dim=2)
        rating = self.final_layer(vector)
        rating = rating.squeeze()
        return rating

    def bpr_loss(self, users, pos, neg):
        user_GMF_emb = self.embedding_GMF_user(users.long())
        pos_GMF_emb = self.embedding_GMF_item(pos.long())
        neg_GMF_emb = self.embedding_GMF_item(neg.long())
        user_MLP_emb = self.embedding_MLP_user(users.long())
        pos_MLP_emb = self.embedding_MLP_item(pos.long())
        neg_MLP_emb = self.embedding_MLP_item(neg.long())
        pos_GMF_vector = torch.mul(user_GMF_emb, pos_GMF_emb) # [batch_size, embed_size]
        neg_GMF_vector = torch.mul(user_GMF_emb, neg_GMF_emb) # [batch_size, embed_size]

        pos_MLP_vector = torch.cat([user_MLP_emb, pos_MLP_emb], dim=-1)
        neg_MLP_vector = torch.cat([user_MLP_emb, neg_MLP_emb], dim=-1)

        pos_MLP_vector = self.MLP_layers(pos_MLP_vector)
        neg_MLP_vector = self.MLP_layers(neg_MLP_vector)

        pos_vector = torch.cat([pos_GMF_vector, pos_MLP_vector], dim=-1)
        neg_vector = torch.cat([neg_GMF_vector, neg_MLP_vector], dim=-1)

        pos_scores = self.final_layer(pos_vector)
        neg_scores = self.final_layer(neg_vector)

        # loss = t.mean(-1.0 * nn.functional.LogSigmoid(neg_scores - pos_scores))
        loss = t.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (user_GMF_emb.norm(2).pow(2) +
                              pos_GMF_emb.norm(2).pow(2) +
                              neg_GMF_emb.norm(2).pow(2) +
                              user_MLP_emb.norm(2).pow(2) +
                              pos_MLP_emb.norm(2).pow(2) +
                              neg_MLP_emb.norm(2).pow(2)) / float(len(users))

        return loss, reg_loss
