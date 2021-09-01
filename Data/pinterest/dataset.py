"""
author: L
date: 2021/8/27 18:24
"""
from tqdm import tqdm
from collections import defaultdict
# test = 'pinterest.test.txt'
# val = 'test.txt'
# with open(test) as f:
#     with open(val, 'w') as a:
#         for line in f.readlines():
#             line = line.strip('\n').split('\t')
#             temp = line[0]
#             temp = temp.split(',')
#             temp[0] = int(temp[0][1:])
#             temp[1] = int(temp[1][:-1])
#             neg_list = [int(i) for i in line[1:]]
#             temp += neg_list
#             for k in temp:
#                 a.write(str(k) + ' ')
#             a.write('\n')

train = 'pinterest.train.txt'
val = 'pinterest.val.txt'
test = 'test.txt'
data = 'data.txt'
count = 0
data_dict = defaultdict(list)
exist_user = []
exist_item = []
user_num, item_num = 0, 0
with open(train) as f:
    for line in tqdm(f.readlines()):
        line = line.strip('\n').split('\t')
        u_id, i_id = int(line[0]), int(line[1])
        data_dict[u_id].append(i_id)
        exist_user.append(u_id)
        exist_item.append(i_id)
        item_num = max(item_num, i_id)
        user_num = max(user_num, u_id)
        count += 1
length_a, length_b = 0, 0
for u_id in data_dict:
    if len(data_dict[u_id]) != len(list(set(data_dict[u_id]))):
        print(u_id, data_dict[u_id])
    length_a += len(data_dict[u_id])
    length_b += len(list(set(data_dict[u_id])))
print(length_a,length_b)

with open(val) as f:
    for line in tqdm(f.readlines()):
        line = line.strip('\n').split('\t')
        u_id, i_id = int(line[0]), int(line[1])
        data_dict[u_id].append(i_id)
        exist_user.append(u_id)
        exist_item.append(i_id)
        item_num = max(item_num, i_id)
        user_num = max(user_num, u_id)
        count += 1

with open(test) as f:
    for line in tqdm(f.readlines()):
        line = line.strip('\n').split(' ')
        u_id, i_id = int(line[0]), int(line[1])
        data_dict[u_id].append(i_id)
        exist_user.append(u_id)
        exist_item.append(i_id)
        item_num = max(item_num, i_id)
        user_num = max(user_num, u_id)
        count += 1
user_num += 1
item_num += 1
print(count)
print(user_num, item_num)
print(len(exist_user), len(exist_item))
print(len(list(set(exist_user))), len(list(set(exist_item))))
length = 0
# with open(data, 'w') as f:
for u_id in data_dict:
    # f.write(str(u_id))
    temp = sorted(list(set(data_dict[u_id])))
    length += len(temp)
        # for item in temp:
        #     f.write(' ' + str(item))
        # f.write('\n')

print(length)