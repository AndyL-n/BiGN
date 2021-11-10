import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from time import time
from tqdm import tqdm
from parse import cprint, args
from sys import exit
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
pd.set_option('display.max_rows',None)

class Loader(Dataset):
    """
    Dataset type for pytorch
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, path="./Data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.path = path
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # get number of users and items
        self.n_user, self.n_item = 0, 0
        self.n_train, self.n_test = 0, 0


        train_unique_users, train_user, train_item = [], [], []
        test_unique_users, test_user, test_item = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    train_user.extend([uid] * len(items))
                    train_item.extend(items)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.n_train += len(items)

        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    self.n_item = max(self.n_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.n_test += len(items)

        self.test_unique_users = np.array(test_unique_users)
        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.n_item += 1
        self.n_user += 1
        
        self.Graph = None
        self.LGraph = None
        self.RGraph = None
        self.similarity = None
        self.social= None

        print(f"{self.n_user} users")
        print(f"{self.n_item} items")
        print(f"{self.n_train} interactions for training")
        print(f"{self.n_test} interactions for testing")
        print(f"{self.path} Sparsity : {(self.n_train + self.n_test) / self.n_user / self.n_item}")

        # [n_user, n_item], bipartite graph
        self.R = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)), shape=(self.n_user, self.n_item))

        # pre-calculate
        self.all_pos = self.get_user_pos(list(range(self.n_user)))
        self.test_dict = self.build_test()
        print(f"{args.dataset} is ready to go🏃")

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):

        print("loading symmetric norm adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/adj_symmetric_mat.npz')
                print("successfully loaded symmetric norm adjacency matrix...")
                norm_adj = pre_adj_mat
            except :
                print("generating symmetric norm adjacency matrix")
                s = time()

                adj_mat = self.adj_mat.todok()
                # L = D^-0.5 * (A + I) * D^-0.5
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                # L = D^-0.5 * A * D^-0.5
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved adj_symmetric_mat...")
                sp.save_npz(self.path + '/adj_symmetric_mat.npz', norm_adj)


            self.Graph = self.convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(args.device)
        return self.Graph

    def getSparseLGraph(self):
        # L = D^-1 * A
        print("loading L norm adjacency matrix")
        if self.LGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/adj_L_mat.npz')
                print("successfully loaded L norm adjacency matrix...")
                norm_adj = pre_adj_mat
                # print(norm_adj)
            except:
                print("generating L norm adjacency  matrix")
                s = time()

                adj_mat = self.adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved adj_L_mat...")
                sp.save_npz(self.path + '/adj_L_mat.npz', norm_adj)


            self.LGraph = self.convert_sp_mat_to_sp_tensor(norm_adj)
            self.LGraph = self.LGraph.coalesce().to(args.device)
            print("don't split the matrix")
        return self.LGraph

    def getSparseRGraph(self):
        # L = A * D^-1
        print("loading R norm adjacency matrix")
        if self.RGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/adj_R_mat.npz')
                print("successfully loaded R norm adjacency matrix...")
                norm_adj = pre_adj_mat
                # print(norm_adj)
            except:
                print("generating R norm adjacency matrix")
                s = time()
                adj_mat = self.adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = adj_mat.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved adj_R_mat...")
                sp.save_npz(self.path + '/adj_R_mat.npz', norm_adj)


            self.RGraph = self.convert_sp_mat_to_sp_tensor(norm_adj)
            self.RGraph = self.RGraph.coalesce().to(args.device)
            print("don't split the matrix")
        return self.RGraph

    def getSimilarity(self):
        print("loading similarity matrix")
        if self.similarity is None:
            if args.neighbor == 10:
                try:
                    similarity = sp.load_npz(self.path + '/similarity_mat.npz')
                    print(similarity[0][0])
                    print("successfully loaded similarity...")
                except:
                    print("generating similarity matrix")
                    s = time()
                    similarity = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
                    similarity = similarity.tolil()
                    R = self.R.tolil()
                    print("generating user similarity")
                    dist_out = 1 - pairwise_distances(R, metric="cosine")
                    similarity[:self.n_user, :self.n_user] = dist_out
                    print("generating item similarity")
                    dist_out = 1 - pairwise_distances(R.T, metric="cosine")
                    similarity[self.n_user:, self.n_user:] = dist_out
                    similarity = similarity.tocsr()
                    end = time()
                    print(f"costing {end - s}s, saved similarity_mat...")
                    sp.save_npz(self.path + '/similarity_mat.npz', similarity)
            else:
                try:
                    similarity = sp.load_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '.npz')
                    print(f"successfully loaded similarity(sample{args.neighbor})...")
                except:
                    try:
                        similarity = sp.load_npz(self.path + '/similarity_mat.npz')
                        print("successfully loaded similarity...")
                    except:
                        print("generating similarity matrix")
                        s = time()
                        similarity = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
                        similarity = similarity.tolil()
                        R = self.R.tolil()
                        print("generating user similarity")
                        dist_out = 1 - pairwise_distances(R, metric="cosine")
                        similarity[:self.n_user, :self.n_user] = dist_out
                        print("generating item similarity")
                        dist_out = 1 - pairwise_distances(R.T, metric="cosine")
                        similarity[self.n_user:, self.n_user:] = dist_out
                        similarity = similarity.tocsr()
                        end = time()
                        print(f"costing {end - s}s, saved similarity_mat...")
                        sp.save_npz(self.path + '/similarity_mat.npz', similarity)

                    # sample
                    row = similarity.shape[0]
                    similarity = similarity.tolil()
                    print("neighbor sampling")
                    for i in tqdm(range(row)):
                        tmp = similarity.getrow(i).toarray()[0]
                        rank_index = tmp.argsort()[::-1]
                        # (i,i) 相似度为 1
                        rank_index = rank_index[:args.neighbor + 1]
                        for j in range(row):
                            if j not in rank_index:
                                tmp[j] = 0
                        similarity[i, :] = tmp
                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '.npz', similarity)

            similarity = similarity.tocsr()
            similarity = self.normalization(similarity)


            self.similarity = self.convert_sp_mat_to_sp_tensor(similarity)
            self.similarity = self.similarity.coalesce().to(args.device)
            print("don't split the matrix")
        return self.similarity

    def getSocial(self):
        print("loading social norm adjacency matrix")
        if self.RGraph is None:
            try:
                social_mat = sp.load_npz(self.path + '/adj_social_mat.npz')
                print("successfully loaded social norm adjacency matrix...")
                social = social_mat
                # print(norm_adj)
            except:
                print("generating social norm adjacency matrix")
                s = time()
                adj_mat = self.adj_mat

                rowsum = np.array(adj_mat.sum(axis=1))  # Du
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv) # Du ^ -0.5
                norm_adj = d_mat.dot(adj_mat)  # Du ^ -0.5 * R

                colsum = np.array(adj_mat.sum(axis=0))  # Di
                d_inv = np.power(colsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv) # Di^-0.5
                norm_adj = norm_adj.dot(d_mat)  # Du ^ -0.5 * R * Di ^ -0.5

                social = norm_adj.T @ norm_adj
                end = time()
                social = social.tocsr()

                print(f"costing {end - s}s, saved adj_social_mat...")
                sp.save_npz(self.path + '/adj_social_mat.npz', social)

            self.social = self.convert_sp_mat_to_sp_tensor(social)
            self.social = self.social.coalesce().to(args.device)
            print("don't split the matrix")
        return self.social

    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_pos(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.R[user].nonzero()[1])
        return pos_items

    def get_user_similarity(self):
        print("loading all user similarity matrix")
        try:
            user_sim = sp.load_npz(self.path + '/user_similarity_mat.npz')
            print("successfully loaded all user similarity matrix...")
        except:
            #item_user 倒排表
            item_users = {}
            for user,item in enumerate(self.all_pos):
                for i in item:
                    if i not in item_users:
                        item_users[i] = set()
                    item_users[i].add(user)

            # 有共同浏览记录的用户的重合度
            # user_mat = sp.dok_matrix((self.n_user, self.n_user), dtype=np.float32)
            # user_mat[0,0] = 1
            # print(user_mat)
            # print(user_mat.shape)
            Sim = {} # C(u,v)
            user_user = {}
            for item, user_set in tqdm(item_users.items()):
                ppl = len(user_set)
                users = sorted(list(user_set))
                for i in range(ppl):

                    Sim[users[i]] = {} if users[i] not in Sim else Sim[users[i]]
                    for j in range(i+1,ppl):
                        tmp = 0 if users[j] not in Sim[users[i]] else Sim[users[i]][users[j]]
                        tmp += 1 / np.log(1 + ppl)
                        Sim[users[i]][users[j]] = tmp
                        # user_mat[int(users[i]),int(users[j])] = tmp
                        # user_mat[int(users[j]),int(users[i])] = tmp

            users, neighbors, scores = [], [], []
            for user,neighbor_users in tqdm(Sim.items()):
                # print(user, neighbor_users)
                for neighbor_user, score in neighbor_users.items():
                    if score != 0:
                        users.append(user)
                        neighbors.append(neighbor_user)
                        scores.append(score)


            tmp = users + neighbors
            neighbors.extend(users)
            users = tmp
            scores = scores * 2
            print(len(users), len(neighbors), len(scores))
            users, neighbors, scores = np.array(users), np.array(neighbors), np.array(scores)
            user_sim = csr_matrix((scores, (users, neighbors)),shape=(self.n_user, self.n_user))
            # self.user_sim = self.user_sim + sp.eye(self.n_user, self.n_user)
            rowsum = np.array(self.R.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            print(len(rowsum))
            user_sim = d_mat.dot(user_sim).dot(d_mat)
            sp.save_npz(self.path + '/user_similarity_mat.npz', user_sim)
            print("successfully generated all user similarity matrix...")
        return user_sim

    def get_item_similarity(self):
        print("loading all item similarity matrix")
        try:
            item_sim = sp.load_npz(self.path + '/item_similarity_mat.npz')
            print("successfully loaded all item similarity matrix...")
        except:
            # item_user 倒排表
            item_users = {}
            Sim = {}  # C(u,v)
            for user, items in enumerate(tqdm(self.all_pos)):
                ppl = len(items)
                for i in range(ppl):
                    Sim[items[i]] = {} if items[i] not in Sim else Sim[items[i]]
                    for j in range(i + 1, ppl):
                        tmp = 0 if items[j] not in Sim[items[i]] else Sim[items[i]][items[j]]
                        tmp += 1 / np.log(1 + ppl)
                        Sim[items[i]][items[j]] = tmp

            items, neighbors, scores = [], [], []
            for item, neighbor_items in tqdm(Sim.items()):
                # print(user, neighbor_users)
                for neighbor_item, score in neighbor_items.items():
                    if score != 0:
                        items.append(item)
                        neighbors.append(neighbor_item)
                        scores.append(score)

            tmp = items + neighbors
            neighbors.extend(items)
            items = tmp
            scores = scores * 2
            print(len(items), len(neighbors), len(scores))
            items, neighbors, scores = np.array(items), np.array(neighbors), np.array(scores)
            item_sim = csr_matrix((scores, (items, neighbors)), shape=(self.n_item, self.n_item))
            # self.user_sim = self.user_sim + sp.eye(self.n_user, self.n_user)
            rowsum = np.array(self.R.T.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            # print(len(rowsum))
            item_sim = d_mat.dot(item_sim).dot(d_mat)
            sp.save_npz(self.path + '/item_similarity_mat.npz', item_sim)
            print("successfully generated all item similarity matrix...")
        return item_sim

dataset = Loader(path="../Data/gowalla")
# dataset.getSimilarity()
# print(dataset.n_user)
# dataset.getSparseGraph()
# # dataset.getSparseRGraph()
# print(dataset.all_pos[0])
# dataset.get_item_similarity()
# print(dataset.social)
# data = sp.load_npz(self.path + '/adj_social_mat.npz')