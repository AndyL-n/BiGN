"""
author: L
date: 2021/8/27 16:05
"""
import random
from tqdm import tqdm
from collections import Counter

files = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
# files = ['ml-100k']
for file in files:
    print("=============Create Dataset================")
    # file = 'ml-100k'
    data = '../Data_all/' + file + '/data.txt'
    rating = file + '/ratings.txt'
    train = file + '/train.txt'
    test = file + '/test.txt'
    val = file + '/val.txt'
    # map = file + '/map.txt'
    item_num = 0
    user_num = 0
    interaction_num = 0
    train_num = 0
    val_num = 0
    test_num = 0
    exist_items = []
    exist_users = []
    neg_sample_list_num = 99
    data_dict = dict()

    print("dataset：" + file)
    with open(data) as f:
        # with open(rating, 'w') as r:
        for line in f.readlines():
            line = [int(i) for i in line.strip('\n').split(' ')]
            u_id = line[0]
            if len(line) > 1 and line[1] != ' ':
                items = line[1:]
            else:
                items = []

            item_num = max(item_num, max(items))
            user_num = max(user_num, u_id)
            interaction_num += len(items)
            # for item in sorted(items):
            #     r.write(str(u_id) + ' ' + str(item) + '\n')
            data_dict[u_id] = items
            exist_items += items
            exist_users.append(u_id)
    user_num, item_num = user_num + 1, item_num + 1
    # print(Counter(exist_items))
    print("user_num：" + str(user_num) + " item_num：" + str(item_num) + " interaction_num：" + str(interaction_num))
    print("exsit_user_num：" + str(len(list(set(exist_users)))) + " exsit_item_num：" + str(len(list(set(exist_items)))))
    print("=============Negative Sample===============")
    with open(train, 'w') as a:
        with open(val, 'w') as b:
            with open(test, 'w') as c:
                # with open(map, 'w') as d:
                    item_list = [i for i in range(item_num)]
                    for u_id in tqdm(data_dict):
                        # interaction_flag = [0 for _ in range(item_num)]
                        pos_list = data_dict[u_id]
                        neg_list = list(set(item_list) - set(pos_list))
                        random.shuffle(pos_list)
                        random.shuffle(neg_list)
                        b.write(str(u_id) + ' ' + str(pos_list[0]) + ' ' + str(random.choice(neg_list)) +'\n')
                        val_num += 1
                        neg_test_list = random.sample(neg_list, neg_sample_list_num)
                        c.write(str(u_id) + ' ' + str(pos_list[1]) + ' ')
                        test_num += 1
                        for neg in neg_test_list:
                            c.write(str(neg) + ' ')
                        c.write('\n')
                        for pos in pos_list[2:]:
                            # interaction_flag[pos] = 1
                            a.write(str(u_id) + ' ' + str(pos) + ' ' + str(random.choice(neg_list)) + '\n')
                        train_num += len(pos_list[2:])
                        # for flag in interaction_flag:
                            # d.write(str(flag) + ' ')
                        # d.write('\n')
                    print("train_num：" + str(train_num) + " val_num：" + str(val_num) + " test_num：" + str(test_num))
                    print("count：" + str(train_num + val_num + test_num))
    print("=============Create Success================")
