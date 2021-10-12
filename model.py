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
import random as rd

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_users_rating(self, users):
        raise NotImplementedError

    def save_model(self, path):
        torch.save(self.state_dict(), path)

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
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})🏃")

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

class DGCF(BasicModel):
    def __init__(self, args, dataset):
        super(DGCF, self).__init__()
        self.args = args
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.args.embed_size
        self.layer = self.args.layer
        self.factor = self.args.factor
        self.iteration = self.args.iteration
        self.split = self.args.split
        self.pick_scale = self.args.pick_scale
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
        self.adj_mat = self.dataset.adj_mat
        self.adj_mat = self.adj_mat.tocoo()
        self.head_list = list(self.adj_mat.row)
        self.tail_list = list(self.adj_mat.col)
        self.val_list = list(self.adj_mat.data)
        self.shape = self.adj_mat.shape

        total_batch = (self.dataset.n_train - 1) // self.args.train_batch + 1
        self.cor_batch_size = int(max(self.n_user/total_batch, self.n_item/total_batch))
        print(f"{self.args.model_name} is already to go(dropout:{self.args.dropout})🏃")

    def disentangling(self, factor_num, factor_values, pick=False):
        A_factors = []
        D_col_factors = []
        D_row_factors = []
        # get the indices of adjacency matrix

        # apply factor-aware softmax function over the values of adjacency matrix
        # ....factor_values is [factor_num, len(val_list)]

        factor_scores = F.softmax(factor_values, 0)

        if pick:
            min_A = torch.min(factor_scores, 0)
            index = factor_scores > (min_A + 0.0000001)
            index = index.type(torch.float32) * (self.pick_scale - 1.0) + 1.0
            # adjust the weight of the minimum factor to 1/self.pick_scale

            factor_scores = factor_scores * index
            factor_scores = factor_scores / torch.sum(factor_scores, 0)

        # 生成每一个intent矩阵
        for i in range(factor_num):
            # in the i-th factor, couple the adjcency values with the adjacency indices
            # .... A i-tensor is a sparse tensor with size of [n_users+n_items,n_users+n_items]
            A_i_scores = factor_scores[i]
            A_i_tensor = torch.sparse_coo_tensor(indices=[self.head_list, self.tail_list], values=A_i_scores,size=self.shape)
            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=1).to_dense())
            D_i_row_scores = 1 / torch.sqrt(torch.sparse.sum(A_i_tensor, dim=0).to_dense())

            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = torch.sparse_coo_tensor(
                indices=[list(range(self.n_user + self.n_item)), list(range(self.n_user + self.n_item))],
                values=D_i_col_scores, size=self.shape)
            D_i_row_tensor = torch.sparse_coo_tensor(
                indices=[list(range(self.n_user + self.n_item)), list(range(self.n_user + self.n_item))],
                values=D_i_row_scores, size=self.shape)

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors

    def computer(self, pick_=False):
        '''
        pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
        pick_ : False, do nothing.
        '''
        p_test = False
        p_train = False

        # get a (n_factors)-length list of [n_users+n_items, n_users+n_items]
        factor_values = torch.ones(self.factor, len(self.val_list)).to(self.args.device)

        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        embs_t = [all_emb]

        output_factors_distribution = []

        # 兴趣因子
        factor_num = [self.factor for _ in range(self.layer)]
        # T 传播距离
        iter_num = [self.iteration for _ in range(self.layer)]
        for l in range(self.layer):
            # prepare the output embedding list
            # .... layer_embeddings stores a (n_factors)-len list of outputs derived from the last routing iterations.
            factor = factor_num[l]
            iteration = iter_num[l]

            layer_emb = []
            layer_emb_t = []

            # split the input embedding table
            # .... ego_layer_emb is a (n_factors)-len list of embeddings [n_users+n_items, embed_size/n_factors]
            # 注意tensorflow 和 torch split 区别
            ego_layer_emb = torch.split(all_emb, self.embed_size//factor, 1)
            ego_layer_emb_t = torch.split(all_emb, self.embed_size//factor, 1)

            # perform routing mechanism
            for t in range(iteration):
                iter_emb = []
                iter_emb_t = []
                iter_val = []

                # split the adjacency values & get three lists of [n_users+n_items, n_users+n_items] sparse tensors
                # .... A_factors is a (n_factors)-len list, each of which is an adjacency matrix
                # .... D_col_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. columns
                # .... D_row_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. rows
                if t == iteration - 1:
                    p_test = pick_
                    p_train = False

                A_factors, D_col_factors, D_row_factors = self.disentangling(factor, factor_values, pick=p_train)
                A_factors_t, D_col_factors_t, D_row_factors_t = self.disentangling(factor, factor_values, pick=p_test)

                # 对intent进行归一化
                for f in range(factor):
                    # 第K个因素 对每个
                    factor_emb = torch.sparse.mm(D_col_factors[f], ego_layer_emb[f])
                    factor_emb = torch.sparse.mm(A_factors[f], factor_emb)
                    factor_emb = torch.sparse.mm(D_col_factors[f], factor_emb)

                    factor_emb_t = torch.sparse.mm(D_col_factors_t[f], ego_layer_emb_t[f])
                    factor_emb_t = torch.sparse.mm(A_factors_t[f], factor_emb_t)
                    factor_emb_t = torch.sparse.mm(D_col_factors_t[f], factor_emb_t)

                    iter_emb.append(factor_emb)
                    iter_emb_t.append(factor_emb_t)

                    if t == iteration - 1:
                        layer_emb = iter_emb
                        layer_emb_t = iter_emb_t

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings

                    # weight
                    head_emb = factor_emb[self.head_list]
                    # distilled information
                    tail_emb = ego_layer_emb[f][self.tail_list]

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_emb = F.normalize(head_emb, dim=1)
                    tail_emb = F.normalize(tail_emb, dim=1)

                    # get the attentive weights
                    # .... factor_value is a dense tensor with the size of [all_h_list,1]
                    factor_value = torch.sum(torch.mul(head_emb, torch.tanh(tail_emb)), axis=1)

                    # update the attentive weights
                    iter_val.append(factor_value)


                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                iter_val = torch.stack(iter_val, 0)
                # add all layer-wise attentive weights up.
                factor_values += iter_val

                if t == iteration - 1:
                    # layer_embeddings = iter_embeddings
                    output_factors_distribution.append(A_factors)

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_emb = torch.cat(layer_emb, 1)
            side_emb_t = torch.cat(layer_emb_t, 1)
            all_emb = side_emb
            all_emb_t = side_emb_t
            # concatenate outputs of all layers
            embs.append(all_emb)
            embs_t.append(all_emb_t)

        embs = torch.stack(embs, 1)
        embs = torch.mean(embs, 1)
        embs_t = torch.stack(embs_t, 1)
        embs_t = torch.mean(embs_t, 1)
        users_emb, items_emb = torch.split(embs, [self.n_user, self.n_item], 0)
        users_emb_t, items_emb_t = torch.split(embs_t, [self.n_user, self.n_item], 0)
        # return users_emb, items_emb, output_factors_distribution, users_emb_t, items_emb_t
        return users_emb, items_emb

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

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = torch.sum(torch.square(X), 1, keepdims=True)
            D = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X, X.t()) + r.t(), 0.0) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - torch.mean(D, dim=0, keepdims=True) - torch.mean(D, dim=1, keepdims=True) \
                + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = D1.shape[0].type(torch.float32)
            dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def create_cor_loss(self,users_emb, items_emb):
        # We have to sample some embedded representations out of all nodes.
        # Becasue we have no way to store cor-distance for each pair.
        cor_users = rd.sample(list(range(self.n_user)), self.cor_batch_size)
        cor_items = rd.sample(list(range(self.n_item)), self.cor_batch_size)

        cor_users = torch.tensor(cor_users, dtype=torch.long)
        cor_items = torch.tensor(cor_items, dtype=torch.long)

        cor_users_emb = users_emb[cor_users.long()]
        cor_items_emb = items_emb[cor_items.long()]
        print(cor_users_emb.shape)

        cor_loss = torch.zeros(1)

        ui_embeddings = torch.cat([cor_users_emb, cor_items_emb], 0)
        print(ui_embeddings)
        ui_factor_embeddings = torch.split(ui_embeddings, self.embed_size//self.factor, 1)
        print(ui_factor_embeddings)

        for i in range(0, self.n_factors - 1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i + 1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors / 2)
        return cor_loss

    def bpr_loss(self, users, pos, neg):
        # (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.get_embedding(users.long(), pos.long(), neg.long())
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        userEmb0 = self.embedding_user(users.long())
        posEmb0 = self.embedding_item(pos.long())
        negEmb0 = self.embedding_item(neg.long())

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
        self.similarity = self.dataset.getSimilarity()
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
            similarity = self.similarity

        # for layer in range(self.layer):
        #     if self.split:
        #         temp_emb = []
        #         for f in range(len(g_droped)):
        #             temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
        #         side_emb = t.cat(temp_emb, dim=0)
        #     else:
        #         side_emb = t.sparse.mm(g_droped, all_emb)
        #
        #     similarity_emb = t.sparse.mm(similarity, all_emb)
        #
        #     attention_side = t.mean((side_emb * all_emb + side_emb), dim=1)
        #     attention_side = t.exp(attention_side)
        #
        #     attention_sim = t.mean((similarity_emb * all_emb + similarity_emb), dim=1)
        #     attention_sim = t.exp(attention_sim)
        #
        #
        #     attention = attention_side + attention_sim
        #
        #     attention_side = attention_side / attention
        #     attention_sim = attention_sim / attention
        #     # print(attention_l)
        #
        #     attention_side = attention_side.unsqueeze(1)
        #     attention_sim = attention_sim.unsqueeze(1)
        #     # print(attention_s)
        #     # print(side_emb)
        #     # print(attention_s*side_emb)
        #     # print(attention_s+attention_l)
        #     # exit()
        #     tmp = all_emb
        #     # all_emb = (attention_side * side_emb) + (attention_sim * similarity_emb)
        #     # print(all_emb.shape)
        #     all_emb = side_emb
        #     # exit()
        #
        #     embs.append(all_emb)
        # embs = t.stack(embs, dim=1)

        for layer in range(self.layer):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(t.sparse.mm(g_droped[f], all_emb))
                side_emb = t.cat(temp_emb, dim=0)
            else:
                side_emb = t.sparse.mm(g_droped, all_emb)

            side_emb = side_emb + all_emb if self.args.residual else side_emb
            similarity_emb = t.sparse.mm(similarity, all_emb)

            attention_side = t.mean((side_emb * all_emb + side_emb), dim=1)
            attention_side = t.exp(attention_side)

            attention_sim = t.mean((similarity_emb * all_emb + similarity_emb), dim=1)
            attention_sim = t.exp(attention_sim)

            attention = attention_side + attention_sim

            attention_side = attention_side / attention
            attention_sim = attention_sim / attention

            attention_side = attention_side.unsqueeze(1)
            attention_sim = attention_sim.unsqueeze(1)

            all_emb = (attention_side * side_emb) + (attention_sim * similarity_emb)
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
        from sys import exit
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = t.cat([users_emb, items_emb])
        # print(all_emb.shape)
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
            # print(side_emb.shape)
            # print(side_L_emb.shape)
            # print((side_emb + side_L_emb).shape)

            attention_s = t.mean((side_emb * all_emb + side_emb) , dim=1)
            # print(attention_s)
            attention_s = t.exp(attention_s)
            # print(attention_s)
            # print(attention_s.shape)

            attention_l = t.mean((side_L_emb * all_emb + side_L_emb) , dim=1)
            # print(attention_l)
            attention_l = t.exp(attention_l)
            # print(attention_l)
            # print(attention_l.shape)

            attention = attention_s + attention_l
            # print(attention)
            # print(attention.shape)

            attention_s = attention_s / attention
            # print(attention_s)
            attention_l = attention_l / attention
            # print(attention_l)


            attention_s = attention_s.unsqueeze(1)
            attention_l = attention_l.unsqueeze(1)
            # print(attention_s)
            # print(side_emb)
            # print(attention_s*side_emb)
            # print(attention_s+attention_l)
            # exit()
            all_emb = all_emb + (attention_s * side_emb) + (attention_l * side_L_emb)
            # print(all_emb.shape)
            # exit()
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
        norm_adj = d_mat.dot(adj_mat)  # Du ^ -0.5 * R  [n_user, n_item]
        colsum = np.array(adj_mat.sum(axis=0))  # Di
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat  # Di^-0.5       [n_item, n_item]
        self.d_mat_i_inv = sp.diags(1 / d_inv)  # Di^0.5        [n_item, n_item]
        norm_adj = norm_adj.dot(d_mat)  # Du ^ -0.5 * R * Di ^ -0.5 [n_user, n_item]

        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, 256)  # 奇异值分解
        # [k, n_user] * [k, k] * [k, n_item]
        end = time.time()
        print('training time for GF-CF', end - start)

    def get_users_rating(self, batch_users):
        norm_adj = self.norm_adj
        adj_mat = self.adj_mat
        batch_test = np.array(adj_mat[batch_users, :].todense())    # [batch, n_item]
        print(batch_test.shape)
        U_2 = batch_test @ norm_adj.T @ norm_adj
        # [batch, n_item] * [n_item, n_user] * [n_user, n_item] = [batch, n_item]
        if (self.args.dataset == 'amazon-book'):
            ret = U_2
        else:
            U_1 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
            # [batch, n_item] * [n_item, n_item] * [n_item, k] * [k, n_item] * [n_item, n_item] = [batch, n_item]
            ret = U_2 + 0.2 * U_1

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
        self.layer = self.args.layer

        self.embedding_GMF_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_GMF_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)

        self.embedding_MLP_user = t.nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size * (2 ** (self.layer - 1)))
        self.embedding_MLP_item = t.nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size * (2 ** (self.layer - 1)))

        nn.init.normal_(self.embedding_GMF_user.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_GMF_item.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_MLP_user.weight, mean=0., std= 0.01)
        nn.init.normal_(self.embedding_MLP_item.weight, mean=0., std= 0.01)
        cprint('use normal initilizer')

        # Layer configuration
        ##  MLP Layers
        MLP_layers = []
        layers_shape = [self.embed_size * (2 ** (self.layer))]
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
        MLP_vector = torch.cat([user_MLP_emb, item_MLP_emb],dim=2)  # [batch_size, n_item, dim*(2^layer)]
        MLP_vector = self.MLP_layers(MLP_vector)    # [batch_size, n_item, dim]

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
