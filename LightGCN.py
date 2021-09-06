import argparse
import os
from os.path import join
import torch as t
from torch import nn, optim
from time import time
import numpy as np
from utils import minibatch, shuffle
import multiprocessing
from parse import args
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



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

class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = t.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = t.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = t.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = t.sum(users_emb * pos_emb, dim=1)
        neg_scores = t.sum(users_emb * neg_emb, dim=1)
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
        scores = t.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = t.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = t.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(t.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(t.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

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
        if self.config['dropout']:
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
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = t.mul(users_emb, items_emb)
        gamma = t.sum(inner_pro, dim=1)
        return gamma




ROOT_PATH = "/Users/gus/Desktop/light-gcn"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m','ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
all_models  = ['mf', 'lgn']

GPU = t.cuda.is_available()
device = t.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2


dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
        Recmodel = recommend_model
        Recmodel.train()
        bpr: utils.BPRLoss = loss_class

        with timer(name="Sample"):
            S = utils.UniformSample_original(dataset)
        users = t.Tensor(S[:, 0]).long()
        posItems = t.Tensor(S[:, 1]).long()
        negItems = t.Tensor(S[:, 2]).long()

        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
        total_batch = len(users) // world.config['bpr_batch_size'] + 1
        aver_loss = 0.
        for (batch_i,
             (batch_users,
              batch_pos,
              batch_neg)) in enumerate(utils.minibatch(users,
                                                       posItems,
                                                       negItems,
                                                       batch_size=world.config['bpr_batch_size'])):
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
            aver_loss += cri
            if world.tensorboard:
                w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
        aver_loss = aver_loss / total_batch
        time_info = timer.dict()
        timer.zero()
        return f"loss{aver_loss:.3f}-{time_info}"

    def test_one_batch(X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg)}

    def Test(dataset, Recmodel, epoch, w=None, multicore=0):
        u_batch_size = world.config['test_u_batch_size']
        dataset: utils.BasicDataset
        testDict: dict = dataset.testDict
        Recmodel: model.LightGCN
        # eval mode with no dropout
        Recmodel = Recmodel.eval()
        max_K = max(world.topks)
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with t.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = t.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = Recmodel.getUsersRating(batch_users_gpu)
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
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
            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            scale = float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if world.tensorboard:
                w.add_scalars(f'Test/Recall@{world.topks}',
                              {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/Precision@{world.topks}',
                              {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                              {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            if multicore == 1:
                pool.close()
            print(results)
            return results

class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def Train(dataset, model, loss_class, epoch, neg_k=1, w=None):
    model = model
    model.train()
    bpr: BPRLoss = loss_class

    with timer(name="Sample"):
        S = UniformSample_original(dataset)
    users = t.Tensor(S[:, 0]).long()
    posItems = t.Tensor(S[:, 1]).long()
    negItems = t.Tensor(S[:, 2]).long()

    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch = len(users) // args.bpr_batch + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            minibatch(users, posItems, negItems, batch_size=args.train_batch)):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"

def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


if __name__ == '__main__':
    print(args)
    np.random.seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(args.seed)
        t.cuda.manual_seed_all(args.seed)
    t.manual_seed(args.seed)
    GPU = t.cuda.is_available()
    device = t.device('cuda' if GPU else "cpu")
    config = {}
    config['bpr_batch_size'] = args.bpr_batch
    config['latent_dim_rec'] = args.recdim
    config['lightGCN_n_layers'] = args.layer
    config['dropout'] = args.dropout
    config['keep_prob'] = args.keepprob
    config['A_n_fold'] = args.a_fold
    config['test_u_batch_size'] = args.testbatch
    config['multicore'] = args.multicore
    config['lr'] = args.lr
    config['decay'] = args.decay
    config['pretrain'] = args.pretrain
    config['A_split'] = False
    config['bigdata'] = False
    dataset = args.dataset
    model = LightGCN()
    model = model.to(device)
    bpr = BPRLoss(model, config)
    Neg_k = 1
    for epoch in range(args.epochs):
        start = time()
        output_information = Train(dataset, model, bpr, epoch, neg_k=Neg_k)
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])