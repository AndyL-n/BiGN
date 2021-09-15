"""
author: L
date: 2021/8/26 14:45
"""

from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from temp.load_data import load_data
import numpy as np
class LightGCN(Model):
    def __init__(self, feature_columns, adj, batch_size, embed_reg=1e-6, decay=1e-4, **kwargs):
        super(LightGCN, self).__init__()
        self.user_fea_col, self.item_fea_col = feature_columns
        # user embedding
        # self.user_embedding = self.add_weight(name='user_embedding',
        #                                       shape=(self.user_fea_col['feat_num'], self.user_fea_col['embed_dim']),
        #                                       initializer=tf.random_normal_initializer(),
        #                                       regularizer=l2(embed_reg),
        #                                       trainable=True,
        #                                       dtype=tf.float32)
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg),
                                        name='user_embedding')
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg),
                                        name='item_embedding')
        # item embedding
        # self.item_embedding = self.add_weight(name='item_embedding',
        #                                       shape=(self.item_fea_col['feat_num'], self.item_fea_col['embed_dim']),
        #                                       initializer=tf.random_normal_initializer(),
        #                                       regularizer=l2(embed_reg),
        #                                       trainable=True,
        #                                       dtype=tf.float32)

        self.dense = Dense(1, activation=None)
        self.decay = 0.001
        self.n_layer = 2
        self.n_fold = 100
        self.batch_size = batch_size

        self.all_user = tf.constant([i for i in range(self.user_fea_col['feat_num'])])
        self.all_item = tf.constant([i for i in range(self.item_fea_col['feat_num'])])

        self.A_fold_hat = self._split_A_hat_node_dropout(adj)

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.user_fea_col['feat_num'] + self.item_fea_col['feat_num']) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.user_fea_col['feat_num'] + self.item_fea_col['feat_num']
            else:
                end = (i_fold + 1) * fold_len

            coo = X[start:end].tocoo().astype(np.float32)
            indices = np.mat([coo.row, coo.col]).transpose()
            temp =  tf.SparseTensor(indices, coo.data, coo.shape)
            A_fold_hat.append(temp)
            # n_nonzero_temp = X[start:end].count_nonzero()
            # A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat
    def call(self, inputs):
        user, pos, neg = inputs
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)
        print(type(user_embed))
        # print(self.user_embedding.trainable)
        # print(self.item_embedding.trainable)
        # ego_embed = tf.concat([self.user_embedding, self.item_embedding], axis=0)
        # all_embed = [ego_embed]
        # # for k in range(0, self.n_layer):
        # #     temp_embed = []
        # #     for f in range(self.n_fold):
        # #         temp_embed.append(tf.sparse.sparse_dense_matmul(self.A_fold_hat[f], ego_embed))
        # #     side_embeddings = tf.concat(temp_embed, 0)
        # #     ego_embeddings = side_embeddings
        # #     all_embed += [ego_embeddings]
        # all_embed = tf.stack(all_embed, 1)
        # all_embed = tf.reduce_mean(all_embed, axis=1, keepdims=False)
        #
        # all_user_embed, all_item_embed = tf.split(all_embed, [self.user_fea_col['feat_num'], self.item_fea_col['feat_num']], 0)
        # user_pre_embed = tf.nn.embedding_lookup(all_user_embed, user, name='user_pre')
        # pos_pre_embed = tf.nn.embedding_lookup(all_item_embed, pos, name='pos_pre')
        # neg_pre_embed = tf.nn.embedding_lookup(all_item_embed, neg, name='neg_pre')
        #
        # # EagerTensor
        # pos_vector = tf.nn.sigmoid(tf.multiply(user_pre_embed, pos_pre_embed))
        # neg_vector = tf.nn.sigmoid(tf.multiply(user_pre_embed, neg_pre_embed))
        # Tensor
        pos_vector = tf.nn.sigmoid(tf.multiply(user_embed, pos_embed))
        neg_vector = tf.nn.sigmoid(tf.multiply(user_embed, neg_embed))
        print(type(pos_vector))
        print(pos_vector.trainable)
        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)

        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2

        # self.add_loss(losses)
        # user_embed = self.user_embedding(user)
        # pos_embed = self.item_embedding(pos)
        # neg_embed = self.item_embedding(neg)

        regularizer = tf.nn.l2_loss(user_embed) + tf.nn.l2_loss(pos_embed) + tf.nn.l2_loss(neg_embed)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer
        losses = losses + emb_loss
        print(losses)
        self.add_loss(losses)

        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user = Input(shape=(1,), dtype=tf.int32)
        pos = Input(shape=(1,), dtype=tf.int32)
        neg = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user, pos, neg], outputs=self.call([user, pos, neg])).summary()

file = 'ml-100k'
embed_dim = 8

feature_columns, train, val, test, adj = load_data(file, embed_dim)
print("======================")
model = LightGCN(feature_columns, adj, batch_size=512)
print("======================")
model.summary()
# model(train)
