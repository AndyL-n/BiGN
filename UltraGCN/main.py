"""
author: L
date: 2021/11/9 15:28
"""
from UltraGCN.dataloader import Loader
from UltraGCN.model import SimGCN
import torch
if __name__ == "__main__":

    params = {}
    params['embed_size'] = 64
    params['name'] = 'SimGCN'
    params['dropout'] = 0
    params['negative_num'] = 10
    params['neighbor_num'] = 10
    dataset = Loader(path="../Data/gowalla")
    model = SimGCN(params, dataset)
    model.get_weight()
    user = torch.tensor([1]).long()
    print(user)
    model.forward()
    # ultragcn = ultragcn.to(params['device'])
    # optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    #
    # train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)
    #
    # print('END')