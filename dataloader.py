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

    def __init__(self, path="Data/gowalla"):
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

        try:
            self.adj_mat = sp.load_npz(self.path + '/adj_mat.npz')
            print("successfully loaded adjacency matrix...")
        except:
            self.adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user + self.n_item), dtype=np.float32)
            self.adj_mat = self.adj_mat.tolil()
            R = self.R.tolil()
            # prevent memory from overflowing
            for i in tqdm(range(5)):
                self.adj_mat[int(self.n_user * i / 5.0):int(self.n_user * (i + 1.0) / 5), self.n_user:] = \
                    R[int(self.n_user * i / 5.0):int(self.n_user * (i + 1.0) / 5)]
                self.adj_mat[self.n_user:, int(self.n_user * i / 5.0):int(self.n_user * (i + 1.0) / 5)] = \
                    R[int(self.n_user * i / 5.0):int(self.n_user * (i + 1.0) / 5)].T
            self.adj_mat = self.adj_mat.tocsr()
            print('already create adjacency matrix', self.adj_mat.shape)
            sp.save_npz(self.path + '/adj_mat.npz', self.adj_mat)

        # degree
        self.users_D = np.array(self.R.sum(axis=1)).squeeze()
        self.items_D = np.array(self.R.sum(axis=0)).squeeze()

        # A + I
        self.users_D[self.users_D == 0.] = 1.
        self.items_D[self.items_D == 0.] = 1.

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

    def normalization(self, similarity):
        E = sp.eye(similarity.shape[0])
        similarity = similarity - E
        print(f"{args.normalization} normalization")
        try:
            similarity = sp.load_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz')
            print(f"successfully loaded similarity(sample{args.neighbor})_{args.normalization}...")
        except:
            s = time()
            if args.normalization == 'symmetric':
                print("generating symmetric normalization")
                adj_mat = csr_matrix((np.array([1 for _ in range(len(similarity.data))]), similarity.nonzero()), shape=similarity.shape)
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                similarity = d_mat.dot(adj_mat)
                similarity = similarity.dot(d_mat)
                similarity.sort_indices()
                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz', similarity)

            elif args.normalization == 'connect_symmetric':
                print("generating connect_symmetric normalization")
                adj_mat = self.adj_mat.tocsr()
                adj_mat = adj_mat + csr_matrix((np.array([1 for _ in range(len(similarity.data))]), similarity.nonzero()),shape=similarity.shape)

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                similarity = norm_adj.tocsr()

                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz',similarity)

            elif args.normalization == 'sotfmax':
                print("generating sotfmax normalization")

                similarity.data = np.exp(similarity.data)
                rowsum = np.array(similarity.sum(axis=1))

                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                d_mat = d_mat.tocsr()

                similarity = d_mat.dot(similarity)
                similarity.sort_indices()
                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz', similarity)

            elif args.normalization == 'min_max':
                print("generating min_max normalization")
                rowmax = np.array(similarity.max(axis=1).data)
                print(rowmax)
                d_inv = np.power(rowmax, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                d_mat = d_mat.tocsr()
                print(d_mat)
                similarity = d_mat.dot(similarity)
                similarity.sort_indices()
                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz', similarity)

            elif args.normalization == 'min_max&sotfmax':
                print("generating min_max&sotfmax normalization")
                rowmax = np.array(similarity.max(axis=1).data)
                d_inv = np.power(rowmax, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                d_mat = d_mat.tocsr()
                similarity = d_mat.dot(similarity)

                similarity.data = np.exp(similarity.data)
                rowsum = np.array(similarity.sum(axis=1))

                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                d_mat = d_mat.tocsr()

                similarity = d_mat.dot(similarity)
                similarity.sort_indices()
                sp.save_npz(self.path + '/similarity_mat_sample_' + str(args.neighbor) + '_' + args.normalization + '.npz', similarity)

            else:
                print("don't normalize the similarity matrix")

            end = time()
            print(f"costing {end - s}s, normalization similarity...")

        return similarity.tocsr()

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

    # def getUserItemFeedback(self, users, items):
    #     """
    #     users:
    #         shape [-1]
    #     items:
    #         shape [-1]
    #     return:
    #         feedback [-1]
    #     """
    #     # print(self.UserItemNet[users, items])
    #     return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def get_user_pos(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.R[user].nonzero()[1])
        return pos_items

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
#
dataset = Loader(path="Data/gowalla")
# dataset.getSimilarity()
# print(dataset.n_user)
# dataset.getSparseGraph()
# # dataset.getSparseRGraph()
# print(dataset.all_pos[0])
# dataset.getSocial()
# data = sp.load_npz(self.path + '/adj_social_mat.npz')