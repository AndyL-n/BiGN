"""
author: L
date: 2021/11/9 15:28
"""

f __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='gowalla_config.ini', type=str, help='config file path')
    args = parser.parse_args()

    print('###################### UltraGCN ######################')


    print('1. Loading Configuration...')
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items = data_param_prepare(args.config_file)
    # exit()
    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)


    ultragcn = UltraGCN(params, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    ultragcn = ultragcn.to(params['device'])
    optimizer = torch.optim.Adam(ultragcn.parameters(), lr=params['lr'])

    train(ultragcn, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params)

    print('END')