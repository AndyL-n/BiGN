# """
# author: L
# date: 2021/10/25 9:25
# """
#
# a = [1,2,3]
# b = [4,5,6]
# a *= 2
# print(a)
# import matplotlib.pyplot as plt
# import csv
# import pandas as pd
# from scipy.sparse import csr_matrix
# import numpy as np
# from collections import Counter
#
# # n10 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor10_2021-10-19.csv', index_col=False)
# # n12 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor12_2021-10-22.csv', index_col=False)
# # n15 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor15_2021-10-08.csv', index_col=False)
# # n20 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor20_2021-10-08.csv', index_col=False)
# # n10 = [float(i.split('[')[1].split(']')[0]) for i in n10['recall']][:900]
# # print(len(n10))
# # n12 = [float(i.split('[')[1].split(']')[0]) for i in n12['recall']][:900]
# # n15 = [float(i.split('[')[1].split(']')[0]) for i in n15['recall']][:900]
# # n20 = [float(i.split('[')[1].split(']')[0]) for i in n20['recall']][:900]
# # plt.grid(alpha=0.3)
# # x = [i for i in range(0,900)]
# # plt.plot(x, n10, 'b.-', label='neighbor10', markevery=[i for i in range(0,900,20)])
# # plt.plot(x, n12, 'r*-', label='neighbor12', markevery=[i for i in range(0,900,20)])
# # plt.plot(x, n15, 'yx-', label='neighbor15', markevery=[i for i in range(0,900,20)])
# # plt.plot(x, n20, 'c,-', label='neighbor20', markevery=[i for i in range(0,900,20)])
# # plt.xlabel('epochs')
# # plt.ylabel('Recall@20')
# # plt.legend(loc='lower right', fontsize='15')
# # plt.show()
#
# users = []
# idx = [[],[]]
# n_user, n_item, n_data = 0, 0, 0
# data = 'Data_all/gowalla/data.txt'
# with open(data) as f:
#     for l in f.readlines():
#         if len(l) > 0:
#             l = l.strip('\n').split(' ')
#             items = [int(i) for i in l[1:]]
#             uid = int(l[0])
#             users.append(uid)
#             idx[0].extend([uid] * len(items))
#             idx[1].extend(items)
#             n_item = max(n_item, max(items))
#             n_user = max(n_user, uid)
#             n_data += len(items)
#
# n_user, n_item = n_user + 1, n_item + 1
# print(max(idx[0]), max(idx[1]))
# print(n_user, n_item, n_data)
# R = csr_matrix((np.ones(n_data), (idx[0], idx[1])), shape=(n_user, n_item))
#
# users_D = np.array(R.sum(axis=1)).squeeze()
# items_D = np.array(R.sum(axis=0)).squeeze()
# x = [i for i in range(n_item)]
# y = sorted(items_D, reverse=True)
# # y = list(Counter(users_D).values())
# # x = list(Counter(users_D).keys())
# # print(x)
# # print(y)
# plt.scatter(x, y, s=4, c='blue', marker='*', alpha=1)
# plt.xlabel('item')
# plt.ylabel('Interactions')
# # plt.legend(loc='upper right', fontsize='15')
# plt.show()
#
# # y = list(Counter(items_D).values())
# # x = list(Counter(items_D).keys())
# # print(x)
# # print(y)
# # plt.scatter(x, y, c='red', alpha=0.5)
# # plt.xlabel('Interactions')
# # plt.ylabel('Recall@20')
# # plt.legend(loc='lower right', fontsize='15')
# # plt.show()

import numpy as np
from sklearn.metrics import pairwise_distances
# A = np.empty([3,3], dtype = int)
# print(A)
# K = 1 # Select top K values along each column
#
# # Sort A, store the argsort for later usage
# sidx = np.argsort(A,axis=0)
# sA = A[sidx,np.arange(A.shape[1])]
#
# # Perform differentiation along rows and look for non-zero differentiations
# df = np.diff(sA,axis=0)!=0
#
# # Perform cumulative summation along rows from bottom upwards.
# # Thus, summations < K should give us a mask of valid ones that are to
# # be kept per column. Use this mask to set rest as zeros in sorted array.
# mask = (df[::-1].cumsum(0)<K)[::-1]
# sA[:-1] *=mask
#
# # Finally revert back to unsorted order by using sorted indices sidx
# out = sA[sidx.argsort(0),np.arange(sA.shape[1])]
# print(out)
# print(out.T)
# R = np.random.rand(2,2)
# R = [[10,2,3],[7,8,9],[1,5,4]]
# k = 2
# # dist_out = 1 - pairwise_distances(R, metric="cosine")
# dist_out = np.array(R)
# print(dist_out)
# print("-------------------")
# k = 2
# sidx = np.argsort(dist_out, axis=1)[:,-2]
# row = np.array([i for i in range(k)])
# print(row)
# print(sidx)
# print(dist_out[1,1])
# print("-------------------")
# sA = dist_out[sidx, np.arange(dist_out.shape[1])]
# print(sA)
# print("-------------------")
# df = np.diff(sA, axis=0) != 0
# print(sA)
# print("-------------------")
# mask = (df[::-1].cumsum(0) < k)[::-1]
# print(mask)
# print("-------------------")
# sA[:-1] *= mask
# print(sA)
# print("-------------------")
# dist_out = sA[sidx.argsort(0), np.arange(sA.shape[1])]
# print(dist_out)
# print("-------------------")
# print(dist_out)
# # dist_out = dist_out.T
# # print(dist_out)

R = np.array([1,3,4,5])
print(R[0,2])