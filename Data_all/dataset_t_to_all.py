"""
author: L
date: 2021/8/27 16:05
"""

files = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
for file in files:
    # file = 'ml-100k'
    train_file = file + '/train.txt'
    test_file = file + '/test.txt'
    data = file + '/data.txt'
    user_num, item_num = 0, 0
    n_train, n_test = 0, 0
    n_count = 0
    data_dict, test_dict = dict(), dict()
    exist_items = []
    with open(train_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                if len(line) > 1 and line[1] != '':
                    items = [int(i) for i in line[1:]]
                else:
                    items = []
                items = list(set(items))
                if items != []:
                    item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                data_dict[u_id] = items
                n_train += len(items)
                exist_items += items
        f.close()

    with open(test_file) as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split(' ')
                u_id = int(line[0])
                if len(line) > 1 and line[1] != '':
                    items = [int(i) for i in line[1:]]
                # else:
                #     print(line)
                else:
                    items = []
                items = list(set(items))
                if items != []:
                    item_num = max(item_num, max(items))
                user_num = max(user_num, u_id)
                data_dict[u_id] = list(items + data_dict[u_id])
                n_count += len(data_dict[u_id])
                n_test += len(items)
                exist_items += items
        f.close()

    user_num += 1
    item_num += 1

    print(user_num, item_num, n_train, n_test, n_count)

    with open(data, 'w') as f:
        count = 0
        for index, u_id in enumerate(data_dict):
            f.write(str(u_id))
            for item in sorted(data_dict[u_id]):
                f.write(' ' + str(item))
                count += 1
            f.write('\n')
        print(count)
