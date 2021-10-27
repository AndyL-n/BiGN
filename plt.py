"""
author: L
date: 2021/10/25 9:25
"""

import matplotlib.pyplot as plt
import csv
import pandas as pd

n10 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor10_2021-10-19.csv', index_col=False)
n12 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor12_2021-10-22.csv', index_col=False)
n15 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor15_2021-10-08.csv', index_col=False)
n20 = pd.read_csv('log/BiGN_gowalla_layer3_dim64_batch2048_K[20]_lr0.001_neighbor20_2021-10-08.csv', index_col=False)
n10 = [float(i.split('[')[1].split(']')[0]) for i in n10['recall']][:900]
print(len(n10))
n12 = [float(i.split('[')[1].split(']')[0]) for i in n12['recall']][:900]
n15 = [float(i.split('[')[1].split(']')[0]) for i in n15['recall']][:900]
n20 = [float(i.split('[')[1].split(']')[0]) for i in n20['recall']][:900]
plt.grid(alpha=0.3)
x = [i for i in range(0,900)]
plt.plot(x, n10, 'b.-', label='neighbor10', markevery=[i for i in range(0,900,20)])
plt.plot(x, n12, 'r*-', label='neighbor12', markevery=[i for i in range(0,900,20)])
plt.plot(x, n15, 'yx-', label='neighbor15', markevery=[i for i in range(0,900,20)])
plt.plot(x, n20, 'c,-', label='neighbor20', markevery=[i for i in range(0,900,20)])
plt.xlabel('epochs')
plt.ylabel('Recall@20')
plt.legend(loc='lower right', fontsize='15')
plt.show()