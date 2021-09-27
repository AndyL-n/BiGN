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
    # 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation', SIGIR2020
    'LightGCN': model.LightGCN,
    # 'Deep Graph Convolutional Networks with Hybrid Normalization for Accurate and Diverse Recommendation', DLP-KDD2021
    'DGCN_HN':model.DGCN_HN,
    # 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018
    'GCN': model.GCN,
    # 'Graph Convolutional Matrix Completion', KDD2018
    'GCMC': model.GCMC,
    # 'Neural Graph Collaborative Filtering', SIGIR2019
    'NGCF':model.NGCF,
    # 'Neural Collaborative Filtering', WWW2017
    'NCF': model.NeuMF,
    # 'Learning deep structured semantic models for web search using clickthrough data', 2013
    # 'TT': model.Two_tower,
    # 'BPR: Bayesian personalized ranking from implicit feedback', 2009
    'BPRMF': model.BPRMF,
    # 'Collaborative filtering with graph information: Consistency and scalable methods' NIPS2018
    # 'GRMF': None,
    # 'Collaborative filtering with graph information: Consistency and scalable methods' NIPS2018
    # 'GRMF-norm': None,
    # 'Multi-Graph Convolution Collaborative Filtering', ICDM2019
    # 'Multi-GCCF': None,
    # 'Graph Convolutional Neural Networks for Web-Scale Recommender Systems', SIGKDD2018
    # 'PinSage': None,
    # 'Deepgcns: Can gcns go as deep as cnns?', IEEE2019
    # 'ResGCN': None,
    # 'Self-supervised Graph Learning for Recommendation', SIGIR2021
    # 'SGL-ED': None,
    # 'Variational Autoencoders for Collaborative Filtering', WWW2018
    # 'Mult-VAE': None,
    # 'Disentangled Graph Collaborative Filtering', SIGIR2020
    # 'DGCF': None,
    # 'Learning Disentangled Representations for Recommendation', NIPS2019
    # 'DisenGCN':None,
    # 'Collaborative Filtering with Graph Information: Consistency and Scalable Methods', NIPS2015
    # 'MacridVAE': None,
    # 'How Powerful is Graph Convolution for Recommendation', CIKM2021
    'GF_CF': model.GF_CF,
    'LGCN_IDE': model.LGCN_IDE,
}