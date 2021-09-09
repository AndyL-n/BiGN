# [batch_size,1,embed_dim] * [batch_szie,num,embed_dim]=[batch_size,num]

import torch as t
import numpy as np
# a = t.tensor([1,2], dtype=t.int)
# b = t.tensor([1,2], dtype=t.int)
#
# print(a.shape)
# print(a)
# print(b.shape)
# print(b)
# c = t.mul(a,b)
# print(c)
# print(c.shape)

# print([2]*10)
# a = [1,2,3,4]
# b = [3,4,5]
# print(b[1:2])
#
# print(len(eval('[8,16,32]')))
#
# def cprint(words : str):
#     print(f"\033[0;30;43m{words}\033[0m")
#
# cprint('aaa')
# result = {'precision': np.array([0.00018755]), 'recall': np.array([0.00053749]), 'ndcg': np.array([0.00040836])}
# a,b,c = result
# precision, recall, ndcg = [result[x] for x in result]
# # print(lambda x: result[x] in result)
# print(precision, recall, ndcg)
# import torch as t
# a = t.Tensor([[1,2],[3,4]])
# print(a)
# print(a.T)
# O = t.zeros([2, 2])
# print(O)
# a = t.cat([t.cat([a, O], dim=1), t.cat([O, a.T], dim=1)], dim=0)
# print(a)
import scipy.sparse as sp
a = np.array([1,2,3,4])
a = sp.diags(a)
a = a.tocsr()
print(a)
b = [[11,12,13,14],[0,0,0,24],[0,0,0,34],[0,0,0,44]]
c = sp.dok_matrix((4,4), dtype=int)
c = c.tolil()
c[:,:] = b
c = c.tocsr()
print(a[0,0])
print("-------------------------")
print(c[0,0])
print("-------------------------")
print(c.dot(a))
print("----------c---------------")
print(a.dot(c))
print("----------a--------------")
print(c*a)
c = c*a
print("-------------------------")
c.sort_indices()
for i in range(c.shape[0]):

    print(c.getrow(i))
    print("-------------------------")