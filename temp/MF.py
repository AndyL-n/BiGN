"""
author: L
date: 2021/8/26 15:59
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class MF(Model):
    def __init__(self, feature_columns, embed_reg=1e-4, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.user_fea_col, self.item_fea_col = feature_columns
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
        self.dense = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform',name = 'prediction')

    def call(self, inputs):
        user, pos, neg = inputs
        user_embed = self.user_embedding(user)
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)

        pos_vector = tf.nn.sigmoid(tf.multiply(user_embed, pos_embed))
        neg_vector = tf.nn.sigmoid(tf.multiply(user_embed, neg_embed))

        # result
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)

        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user = Input(shape=(1,), dtype=tf.int32, name='user_id')
        pos = Input(shape=(1,), dtype=tf.int32, name='pos_item_id')
        neg = Input(shape=(1,), dtype=tf.int32, name='neg_item_id')
        Model(inputs=[user, pos, neg], outputs=self.call([user, pos, neg])).summary()


def test():
    model = MF([{'feat': 'user', 'feat_num': 10, 'embed_dim': 8},\
                {'feat': 'item', 'feat_num': 10, 'embed_dim': 8}])
    model.summary()

test()