import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Embedding, Input
from tensorflow.keras.regularizers import l2
n_layer = 1
n_user = 100
n_item = 100
dim = 64
initial_u = tf.random_normal_initializer()
initial_v = tf.random_uniform_initializer()
diagonal=[1.0 for i in range(n_user + n_item)]
l_adj, s_adj = tf.linalg.diag(diagonal), tf.linalg.diag(diagonal)
user_embedding = tf.Variable(initial_u(shape=(n_user, dim)), dtype='float32')
item_embedding = tf.Variable(initial_v(shape=(n_user, dim)), dtype='float32')


all_emb = tf.concat([user_embedding, item_embedding], axis=0)
embs = [all_emb]
for layer in range(n_layer):
    s_emb = tf.matmul(s_adj, all_emb)
    l_emb = tf.matmul(l_adj, all_emb)
    z_s = s_emb + tf.multiply(s_emb, all_emb)
    z_l = l_emb + tf.multiply(l_emb, all_emb)
    z_s = tf.reduce_mean(s_emb,axis=-1)
    z_l = tf.reduce_mean(s_emb,axis=-1)
    z_s = tf.expand_dims(z_s, 1)
    z_l = tf.expand_dims(z_l, 1)
    attention = tf.concat([z_s,z_l], axis=-1)
    attention = tf.nn.softmax(attention)
    z_s, z_l = tf.split(attention, [1,1], axis=-1)
    all_emb = all_emb + tf.multiply(z_s, s_emb) + tf.multiply(z_l, l_emb)
    embs.append(all_emb)
embs = tf.stack(embs, axis=1)
# print(embs.size())
output = tf.reduce_mean(embs, axis=1)
user, item = tf.split(output, [n_user, n_item], axis=0)