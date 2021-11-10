# """
# author: L
# date: 2021/10/25 9:25
# """
#
a = [1,2,3]
b = [4,5,6]
a *= 2
print(a)
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