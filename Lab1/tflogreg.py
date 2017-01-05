import tensorflow as tf
import numpy as np


class TFLogReg:
    def __init__(self, d, c, param_delta=0.5):
        self.x = tf.placeholder(tf.float32, shape=[None, d])
        self.yoh_ = tf.placeholder(tf.float32, shape=[None, c])
        self.w = tf.Variable(tf.random_normal([d, c]), name='w')
        self.b = tf.Variable(tf.zeros([c]), name='b')
        self.probs = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.probs * self.yoh_, 1))) / np.log(2)
        self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(loss=self.loss)
        self.session = tf.Session()

    def train(self, x, yoh_, param_niter):
        self.session.run(tf.initialize_all_variables())
        for i in range(param_niter + 1):
            loss, step, w, b = self.session.run([self.loss, self.train_step, self.w, self.b],
                                                feed_dict={self.x: x, self.yoh_: yoh_})
            if i % 1000 == 0:
                print("iteration #{}:".format(i), loss, w, b, sep='\n')

    def eval(self, x):
        return self.session.run(self.probs, feed_dict={self.x: x})
