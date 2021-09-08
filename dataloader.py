import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
from parse import cprint, args
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, path="Data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = args.split
        self.folds = args.a_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None
        self.user_sim_mat = None
        self.item_sim_mat = None
        print(f"{self.n_user} users")
        print(f"{self.m_item} items")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{args.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(args.device)
                print("don't split the matrix")
        return self.Graph

    # def coo_cosine_similarity(input_coo_matrix):
    #     sq = lambda x: x * x.T
    #     output_csr_matrix = input_coo_matrix.tocsr()
    #     sqrt_sum_square_rows = np.array(np.sqrt(sq(output_csr_matrix).sum(axis=1)))[:, 0]
    #     output_csr_matrix.data /= rows_sums_sqrt[input_coo_matrix.row]
    #     return sq(output_csr_matrix)

    def getSimGraph(self):
        print("loading similarity matrix")
        if self.user_sim_mat is None:
            try:
                user_sim_mat = sp.load_npz(self.path + '/user_sim_mat.npz')
                print("successfully loaded user_sim_mat...")
            except :
                print("generating user similarity matrix")
                s = time()
                user_sim_mat = sp.dok_matrix((self.n_users, self.n_users), dtype=np.float32)
                user_sim_mat = user_sim_mat.tolil()
                dist_out = 1 - pairwise_distances(self.UserItemNet, metric="cosine")
                user_sim_mat[:] = dist_out
                user_sim_mat = user_sim_mat.tocsr()
                end = time()
                print(f"costing {end-s}s, saved user_sim_mat...")
                sp.save_npz(self.path + '/user_sim_mat.npz', user_sim_mat)

            if self.user_sim_mat is None:
                try:
                    item_sim_mat = sp.load_npz(self.path + '/item_sim_mat.npz')
                    print("successfully loaded user_sim_mat...")
                except :
                    print("generating item similarity matrix")
                    s = time()
                    item_sim_mat = sp.dok_matrix((self.m_items, self.m_items), dtype=np.float32)
                    item_sim_mat = item_sim_mat.tolil()
                    dist_out = 1 - pairwise_distances(self.UserItemNet.T, metric="cosine")
                    item_sim_mat[:] = dist_out
                    print(dist_out.shape)
                    item_sim_mat = item_sim_mat.tocsr()
                    end = time()
                    print(f"costing {end - s}s, saved item_sim_mat...")
                    sp.save_npz(self.path + '/item_sim_mat.npz', item_sim_mat)

            if self.split == True:
                self.user_sim_mat = self._split_A_hat(user_sim_mat)
                self.item_sim_mat = self._split_A_hat(item_sim_mat)
                print("done split matrix")
            else:
                self.user_sim_mat = self._convert_sp_mat_to_sp_tensor(user_sim_mat)
                self.user_sim_mat = self.user_sim_mat.coalesce().to(args.device)
                self.item_sim_mat = self._convert_sp_mat_to_sp_tensor(item_sim_mat)
                self.item_sim_mat = self.item_sim_mat.coalesce().to(args.device)
                print("don't split the matrix")
        return self.user_sim_mat, self.item_sim_mat

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

dataset = Loader(path="Data/"+args.dataset)
dataset.getSimGraph()

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