from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

import numpy as np
import tensorflow as tf


class NNValueFunction(object):
    def __init__(self, obs_dim, hid_list):
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epochs = 10
        self.lr = None
        self.hid_list = hid_list
        self._build_graph()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(
                tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            if self.hid_list == []:
                hid1_size = self.obs_dim * 10  # 10 chosen empirically on 'Hopper-v1'
                hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
                hid2_size = int(np.sqrt(hid1_size * hid3_size))
            else:
                hid1_size = self.hid_list[0]
                hid2_size = self.hid_list[1]
                hid3_size = self.hid_list[2]
            self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid1_size)), name="h2")
            out = tf.layers.dense(out, hid3_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid2_size)), name="h3")
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / hid3_size)), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(
                tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)


    def fit(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for _ in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        np.mean(np.square(y_hat - y))     
  

    def predict(self, x):
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)
        return np.squeeze(y_hat)
    
    def close_sess(self):
        self.sess.close()
