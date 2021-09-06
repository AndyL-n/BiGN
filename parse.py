"""
author: L
date: 2021/9/6 15:58
"""
import argparse
import torch as t
def parse_args():
    parser = argparse.ArgumentParser(description="Run BiGN.")
    parser.add_argument('--train_batch', type=int,default=2048,
                        help="The batch size for bpr loss training procedure.")
    parser.add_argument('--test_batch', type=int, default=1024,
                        help='The batch size of test.')
    parser.add_argument('--embed_size', type=int,default=64,
                        help="Embedding size.")
    parser.add_argument('--layer', type=int,default=3,
                        help="The layer num of BiGN.")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="Learning Rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="Regularizations.")
    parser.add_argument('--dropout', type=int,default=0,
                        help="Using the dropout or not.")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="The batch size for bpr loss training procedure.")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn",
                        help="Comment.")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000,
                        help="The number of epochs.")
    parser.add_argument('--multicore', type=int, default=0,
                        help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--model', type=str, default='lgn',
                        help='rec-model, support [lgn]')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=20)
    GPU = t.cuda.is_available()
    device = t.device('cuda' if GPU else "cpu")
    parser.add_argument('--device', default=device)
    # parser.add_argument('--act', type=str, default="leakyrelu")
    return parser.parse_args()


args = parse_args()
print(args.device)
