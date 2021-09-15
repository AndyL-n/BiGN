"""
author: L
date: 2021/8/24 13:54
"""

from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, Input
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class DNN(Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
		"""
		DNN part
		:param hidden_units: A list. List of hidden layer units's numbers
		:param activation: A string. Activation function
		:param dnn_dropout: A scalar. dropout number
		"""
		super(DNN, self).__init__(**kwargs)
		self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
		self.dropout = Dropout(dnn_dropout)

	def call(self, inputs, **kwargs):
		x = inputs
		for dnn in self.dnn_network:
			x = dnn(x)
		x = self.dropout(x)
		return x

class DSSM(Model):
	def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
		"""
		DSSM model
		:param feature_columns: A list. user feature columns + item feature columns
		:param hidden_units: A list.
		:param dropout: A scalar.
		:param activation: A string.
		:param embed_reg: A scalar. The regularizer of embedding.
		"""
		super(DSSM, self).__init__(**kwargs)
		if hidden_units is None:
			hidden_units = [64, 32, 16, 8]

		self.user_fea_col, self.item_fea_col = feature_columns
		# user embedding
		self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
										   input_length=1,
										   output_dim=self.user_fea_col['embed_dim'],
										   embeddings_initializer='random_normal',
										   embeddings_regularizer=l2(embed_reg),
										   name='user_embedding')
		# item embedding
		self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
										   input_length=1,
										   output_dim=self.item_fea_col['embed_dim'],
										   embeddings_initializer='random_normal',
										   embeddings_regularizer=l2(embed_reg),
										   name='item_embedding')

		self.user_dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
		self.item_dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
		self.dense = Dense(1, activation=None)
		self.decay = 0.001

	def call(self, inputs):
		user, pos, neg = inputs
		user_embed = self.user_embedding(user)
		pos_embed = self.item_embedding(pos)
		neg_embed = self.item_embedding(neg)

		user_vector = self.user_dnn(user_embed)
		pos_vector = self.item_dnn(pos_embed)
		neg_vector = self.item_dnn(neg_embed)

		pos_vector = tf.nn.sigmoid(tf.multiply(user_vector, pos_vector))
		neg_vector = tf.nn.sigmoid(tf.multiply(user_vector, neg_vector))

		# result
		pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)  # (None, 1)
		neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)  # (None, 1/101)
		# loss
		losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
								tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2

		# regularizer
		regularizer = tf.nn.l2_loss(user_embed) + tf.nn.l2_loss(pos_embed) + tf.nn.l2_loss(neg_embed)
		emb_loss = self.decay * tf.reduce_mean(regularizer)
		losses = emb_loss + losses

		self.add_loss(losses)
		logits = tf.concat([pos_logits, neg_logits], axis=-1)
		return logits

	def summary(self):
		user = Input(shape=(1,), dtype=tf.int32)
		pos = Input(shape=(1,), dtype=tf.int32)
		neg = Input(shape=(1,), dtype=tf.int32)
		Model(inputs=[user, pos, neg],outputs=self.call([user, pos, neg])).summary()

