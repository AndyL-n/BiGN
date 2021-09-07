import model
from pprint import pprint
import sys
from dataloader import Loader
from parse import args

all_dataset = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m','ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
if args.dataset in all_dataset:
    dataset = Loader(path="data/"+args.dataset)
else:
    sys.exit("No such file or directory:" + args.dataset)
print('===========config================')
pprint(args)
# # print("cores for test:", world.CORES)
# # print("comment:", world.comment)
# # print("tensorboard:", world.tensorboard)
# # print("LOAD:", world.LOAD)
# # print("Weight path:", world.PATH)
# # print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')
#
MODELS = {
    'lgn': model.LightGCN
}