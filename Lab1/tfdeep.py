import tensorflow as tf
import numpy as np


class TFDeep:
    def __init__(self, ds, param_delta=0.5, param_lambda=0., stddev=1.):
        d = ds[0]
        c = ds[-1]
        self.x = tf.placeholder(tf.float32, shape=[None, d])
        self.yoh_ = tf.placeholder(tf.float32, shape=[None, c])
        # self.ws = [tf.Variable(tf.random_uniform([ds[i - 1], ds[i]], 1e-3, 1e-2), name='W' + str(i)) for i in
        #            range(1, len(ds))]
        self.ws = [tf.Variable(tf.random_normal([ds[i - 1], ds[i]], stddev=stddev), name='W' + str(i)) for i in
                   range(1, len(ds))]
        self.bs = [tf.Variable(tf.zeros([ds[i]]), name='b' + str(i)) for i in range(1, len(ds))]
        self.hs = [self.x]
        for i in range(len(self.ws) - 1):
            self.hs.append(tf.nn.relu6(tf.matmul(self.hs[i], self.ws[i]) + self.bs[i]))
        self.probs = tf.nn.softmax(tf.matmul(self.hs[-1], self.ws[-1]) + self.bs[-1])
        regularization = 0
        for w in self.ws:
            regularization += tf.trace(tf.matmul(w, w, transpose_a=True))
        if param_lambda == 0.:
            self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.probs * self.yoh_, 1))) / np.log(2)
        else:
            self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.probs * self.yoh_, 1))) / np.log(
                2) + param_lambda * regularization
        self.train_step = tf.train.AdamOptimizer(param_delta).minimize(loss=self.loss)
        self.session = tf.Session()

    def train(self, x, yoh_, param_niter):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter + 1):
            loss, step, ws, bs = self.session.run([self.loss, self.train_step, self.ws, self.bs],
                                                  feed_dict={self.x: x, self.yoh_: yoh_})
            if i % 10 == 0:
                print("iteration #{}:".format(i), loss, sep='\n')

    def eval(self, x):
        return self.session.run(self.probs, feed_dict={self.x: x})

    def params(self):
        return self.session.run([self.ws, self.bs])
