"""
author: L
date: 2021/11/9 15:30
"""
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

class SimGCN(PairWiseModel):
    def __init__(self, params, dataset):
        super(SimGCN, self).__init__()
        self.params = params
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.n_user = self.dataset.n_user
        self.n_item = self.dataset.n_item
        self.embed_size = self.params['embed_size']
        self.embedding_user = nn.Embedding(num_embeddings=self.n_user, embedding_dim=self.embed_size)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_item, embedding_dim=self.embed_size)
        self.neighbor_num = self.params['neighbor_num']

        self.item_sim_neighbor, self.item_sim_weight = self.get_item_mat()
        self.user_sim_neighbor, self.user_sim_weight = self.get_user_mat()
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseDGraph()
        self.R = self.dataset.getR()
        print(f"{self.params['name']} is already to go(dropout:{self.params['dropout']})🏃")

    def get_item_mat(self):
        weight_mat = torch.zeros((self.n_item, self.neighbor_num))
        neighbor_mat = torch.zeros((self.n_item, self.neighbor_num))
        item_sim = self.dataset.get_item_similarity()
        for i in range(self.n_item):
            row = torch.from_numpy(item_sim.getrow(i).toarray()[0])
            row_sim, row_idx = torch.topk(row, self.neighbor_num)
            weight_mat[i] = row_sim
            neighbor_mat[i] = row_idx

        print(f"successfully sample {self.neighbor_num} item neighbor")
        return neighbor_mat.long(), weight_mat.float()

    def get_user_mat(self):
        weight_mat = torch.zeros((self.n_user, self.neighbor_num))
        neighbor_mat = torch.zeros((self.n_user, self.neighbor_num))
        user_sim = self.dataset.get_user_similarity()
        for i in range(self.n_user):
            row = torch.from_numpy(user_sim.getrow(i).toarray()[0])
            row_sim, row_idx = torch.topk(row, self.neighbor_num)
            weight_mat[i] = row_sim
            neighbor_mat[i] = row_idx

        print(f"successfully sample {self.neighbor_num} user neighbor")
        return neighbor_mat.long(), weight_mat.float()

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        emb = torch.cat([users_emb, items_emb])
        user_neighbor = self.embedding_user(self.user_sim_neighbor) # [n_user, neighor_num, emb_dim]
        user_neighbor_emb = (user_neighbor * self.user_sim_weight.unsqueeze(2)).sum(dim=1)
        item_neighbor = self.embedding_item(self.item_sim_neighbor)
        item_neighbor_emb = (item_neighbor * self.item_sim_weight.unsqueeze(2)).sum(dim=1)
        neighbor_emb = torch.cat([user_neighbor_emb, item_neighbor_emb])
        all_emb = torch.mm(self.Graph, emb)
        emb = emb + neighbor_emb + all_emb
        users, items = t.split(emb, [self.n_user, self.n_item])
        return users, items

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def bpr_loss(self, user, pos, neg):
        user_emb, item_emb = self.computer()
        loss = self.get_loss(user, pos, neg, user_emb, item_emb)
        reg_loss = self.norm_loss()
        loss = loss + self.decay * reg_loss
        return loss

    def get_loss(self, user, pos, neg, user_emb, item_emb):
        device = self.get_device()
        user_embeds = user_emb[user]
        pos_embeds = item_emb[pos]
        neg_embeds = item_emb[neg]
        weight = self.get_weight(user, pos, neg)

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)
        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=weight[:len(pos_scores)],reduction='none')

        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)
        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels,
                                                      weight=weight[len(pos_scores):].view(neg_scores.size()),
                                                      reduction='none').mean(dim=-1)

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def get_weight(self, user, pos, neg):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = self.R[user * self.n_item + pos].to(device)
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos)).to(device)

        user = (user * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = self.R[torch.cat([user] * neg.size(1)).transpose(1, 0) + neg].flatten().to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg.size(0) * neg.size(1)).to(device)

        weight = torch.cat((pos_weight, neg_weight))
        return weight