import torch as t
from dataloader import BasicDataset
from torch import nn
from parse import cprint

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
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
    def __init__(self, args, dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.args.embed_size
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keep_prob
        self.A_split = self.args.split
        self.embedding_user = t.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = t.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
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
        #   t.split(all_emb , [self.num_users, self.num_items])
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
        users, items = t.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
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
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
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
    def __init__(self, args, dataset: BasicDataset):
        super(BiGN, self).__init__()
        self.args = args
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.args.embed_size
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keep_prob
        self.A_split = self.args.split
        self.embedding_user = t.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = t.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
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

        self.Graph = self.dataset.getSimilarity()
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
        #   t.split(all_emb , [self.num_users, self.num_items])
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
        users, items = t.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
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
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
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
    def __init__(self, args, dataset: BasicDataset):
        super(DGCN_HN, self).__init__()
        self.args = args
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.args.embed_size
        self.n_layers = self.args.layer
        self.keep_prob = self.args.keep_prob
        self.A_split = self.args.split
        self.embedding_user = t.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = t.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
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
        #   t.split(all_emb , [self.num_users, self.num_items])
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
            print(side_emb.shape)
            print(side_L_emb.shape)
            attention_s = t.mean((side_emb * all_emb + side_emb) , dim=1)
            attention_l = t.mean((side_L_emb * all_emb + side_L_emb) , dim=1)
            attention = t.exp(attention_s) + t.exp(attention_s)
            attention_s = t.exp(attention_s) / attention
            attention_l = t.exp(attention_l) / attention
            attention_s = attention_s.unsqueeze(1)
            attention_l = attention_l.unsqueeze(1)
            # print(attention_s.shape)
            # print(attention_l.shape)
            all_emb = all_emb + (attention_s * side_emb) + (attention_l * side_L_emb)
            # print(all_emb.shape)
            embs.append(all_emb)
        embs = t.stack(embs, dim=1)
        # print(embs.size())
        light_out = t.mean(embs, dim=1)
        users, items = t.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(t.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
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
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
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