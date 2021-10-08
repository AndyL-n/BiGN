"""
author: L
date: 2021/9/6 15:58
"""
import argparse
import torch as t

def parse_args():
    parser = argparse.ArgumentParser(description="Run BiGN.")
    parser.add_argument('--model_name', type=str, default='BiGN',
                        help="The name of model. support [BiGN, LightGCN, DGCN_HN, GCN, GCMC, NGCF, NeuMF, TT, BPRMF, GF_CF, LGCN_IDE]")
    parser.add_argument('--train_batch', type=int,default=2048,
                        help="The batch size for bpr loss training procedure.")
    parser.add_argument('--test_batch', type=int, default=2048,
                        help='The batch size of test.')
    parser.add_argument('--embed_size', type=int,default=64,
                        help="Embedding size.")
    parser.add_argument('--layer', type=int,default=3,
                        help="The layer num of BiGN.")
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--lr', type=float,default=0.0001,
                        help="Learning Rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="Regularizations.")
    parser.add_argument('--dropout', type=int,default=0,
                        help="Using the dropout or not.")
    parser.add_argument('--keep_prob', type=float,default=0.6,
                        help="Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.")
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--split', type=bool, default=False,
                        help="Using the split or not.")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [gowalla, yelp2018, amazon-book]")
    # parser.add_argument('--path', type=str,default="./checkpoints",
    #                     help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    # parser.add_argument('--tensorboard', type=int,default=1,
    #                     help="enable tensorboard")
    # parser.add_argument('--comment', type=str,default="lgn",
    #                     help="Comment.")
    parser.add_argument('--neighbor', type=int, default=20,
                        help="The number of neighbor.")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="The number of epochs.")
    parser.add_argument('--normalization', type=str, default='connect_symmetric',
                        help='The approach of normalization, support [symmetric, connect_symmetric, L, R, sotfmax, min_max, min_max&sotfmax]')
    parser.add_argument('--residual', type=bool, default=False,
                        help='whether we use residual connection or not')
    parser.add_argument('--pretrain', type=bool, default=False,
                        help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=20)
    GPU = t.cuda.is_available()
    device = t.device('cuda' if GPU else "cpu")
    parser.add_argument('--device', default=device)
    # parser.add_argument('--act', type=str, default="leakyrelu")
    return parser.parse_args()

args = parse_args()
# print(args.device)f

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")