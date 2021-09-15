"""
author: L
date: 2021/8/25 15:02
"""

import os
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.optimizers import Adam
from temp.NCF import NCF
from temp.load_data import load_data
from evaluate import evaluate_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # =============================== GPU ==============================
    gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('gpu:',gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

    # ========================= Hyper Parameters =======================
    files = ['amazon-book', 'gowalla', 'ml-1m', 'ml-10m', 'ml-20m', 'ml-25m', 'ml-100k', 'yelp2018']
    file = 'ml-100k'
    model_name = 'NCF'

    embed_reg = 1e-6  # 1e-6
    K = 20

    embed_dim = 32
    hidden_units = [256, 128, 64]
    embed_reg = 1e-6  # 1e-6
    activation = 'relu'
    dropout = 0.2
    test_neg_num = 100
    learning_rate = 0.001
    epochs = 100
    batch_size = 512

    # ========================== Create dataset =======================
    feature_columns, train, val, test = load_data(file, embed_dim, test_neg_num)
    # ============================Build Model==========================
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = NCF(feature_columns, hidden_units, dropout, activation, embed_reg)
    model.summary()
    # =========================Compile============================
    model.compile(optimizer=Adam(learning_rate=learning_rate))

    results = []
    for epoch in range(1, epochs + 1):
        # ===========================Fit==============================
        t1 = time.time()
        model.fit(train, None, validation_data=(val, None), epochs=1, batch_size=batch_size, )
        # ===========================Test==============================
        t2 = time.time()
        hit_rate, ndcg = evaluate_model(model, test, K)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f' % (epoch, t2 - t1, time.time() - t2, hit_rate, ndcg))
        results.append([epoch, t2 - t1, time.time() - t2, hit_rate, ndcg])
    # ========================== Write Log ===========================
    timestamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    pd.DataFrame(results, columns=['Iteration', 'fit_time', 'evaluate_time', 'hit_rate', 'ndcg'])\
        .to_csv('log/{}_log_{}_dim_{}_K_{}_{}.csv'\
        .format(model_name, file, embed_dim, K, timestamp), index=False)