"""
author: L
date: 2021/8/24 15:12
"""

import numpy as np
import itertools
import numpy as np
import sys
import heapq
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

def evaluate_model(model, test, K=10):
    """
    evaluate model
    :param model: model
    :param test: test set
    :param K: top K
    :return: hit rate, ndcg
    """
    # 评估方案：为了评价项目推荐的性能，采用了leave-one-out方法评估，
    # 即：对于每个用户，我们将其最近的一次交互作为测试集（数据集一般都有时间戳），并利用余下的培训作为训练集。
    # 由于在评估过程中为每个用户排列所有项目花费的时间太多，所以遵循一般的策略，
    # 随机抽取100个不与用户进行交互的项目，将测试项目排列在这100个项目中。
    # 排名列表的性能由命中率（HR）和归一化折扣累积增益（NDCG）来衡量。
    # 同时，将这两个指标的排名列表截断为10。
    # 如此一来，HR直观地衡量测试项目是否存在于前10名列表中，
    # 而NDCG通过将较高分数指定为顶级排名来计算命中的位置。本文计算每个测试用户的这两个指标，并求取了平均分
    pred_y = - model.predict(test)
    rank = pred_y.argsort().argsort()[:, 0]
    hr, ndcg = 0.0, 0.0
    for r in rank:
        if r < K:
            hr += 1
            ndcg += 1 / np.log2(r + 2)
    return hr / len(rank), ndcg / len(rank)
