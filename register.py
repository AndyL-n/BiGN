import model
from pprint import pprint
import sys
from dataloader import Loader
from parse import args

all_dataset = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m','ml-20m', 'ml-25m', 'ml-100k', 'pinterest', 'yelp2018']
if args.dataset in all_dataset:
    dataset = Loader(path="Data/"+args.dataset)
else:
    sys.exit("No such file or directory:" + args.dataset)
print('===========config================')
pprint(args)
print("using bpr loss")
print('===========end===================')
#
MODELS = {
    'BiGN': model.BiGN,
    'LightGCN': model.LightGCN, # 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation', SIGIR2020
    'DGCN_HN':model.DGCN_HN,    # 'Deep Graph Convolutional Networks with Hybrid Normalization for Accurate and Diverse Recommendation', DLP-KDD2021
    'GCN': model.GCN,           # 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018
    'GCMC': model.GCMC,         # 'Graph Convolutional Matrix Completion', KDD2018
    'NGCF':model.NGCF,          # 'Neural Graph Collaborative Filtering', SIGIR2019
    'NCF': model.NeuMF,         # 'Neural Collaborative Filtering', WWW2017
    'TT': model.Two_tower,      # 'Learning deep structured semantic models for web search using clickthrough data', 2013
    'BPRMF': model.BPRMF,       # 'BPR: Bayesian personalized ranking from implicit feedback', 2009
    'GRMF': None,               # 'Collaborative filtering with graph information: Consistency and scalable methods' NIPS2018
    'GRMF-norm': None,          # 'Collaborative filtering with graph information: Consistency and scalable methods' NIPS2018
    'Multi-GCCF': None,         # 'Multi-Graph Convolution Collaborative Filtering', ICDM2019
    'PinSage': None,            # 'Graph Convolutional Neural Networks for Web-Scale Recommender Systems', SIGKDD2018
    'ResGCN': None,             # 'Deepgcns: Can gcns go as deep as cnns?', IEEE2019
    'SGL-ED': None,             # 'Self-supervised Graph Learning for Recommendation', SIGIR2021
    'Mult-VAE': None,           # 'Variational Autoencoders for Collaborative Filtering', WWW2018
    'DGCF': None,               # 'Disentangled Graph Collaborative Filtering', SIGIR2020
    'DisenGCN':None,            # 'Learning Disentangled Representations for Recommendation', NIPS2019
    'MacridVAE': None           # 'Collaborative Filtering with Graph Information: Consistency and Scalable Methods', NIPS2015
}