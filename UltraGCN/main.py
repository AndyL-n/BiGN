"""
author: L
date: 2021/11/9 15:28
"""
from UltraGCN.dataloader import dataset
from UltraGCN.model import UltraGCN
if __name__ == "__main__":

    params = {}
    params['embed_size'] = 64
    params['name'] = 'UltraGCN'
    params['dropout'] = 0
    model = UltraGCN(params, dataset)
    # ultragcn = ultragcn.to(params['device'])
    # optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])
    #
    # train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)
    #
    # print('END')