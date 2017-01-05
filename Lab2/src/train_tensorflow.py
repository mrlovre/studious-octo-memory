from tensorflow.examples.tutorials.mnist import input_data

import nn
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from attrdict import attrdict

DATA_DIR = '/home/frka/python/Lab2/datasets/MNIST/'
SAVE_DIR = '/home/frka/python/Lab2/out4/'


dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 1, 28, 28])
train_y = dataset.train.labels
train_mean = train_x.mean()
train_x -= train_mean
train_x = train_x.transpose(0, 2, 3, 1)
# valid_x = dataset.validation.images
# valid_x = valid_x.reshape([-1, 1, 28, 28]).transpose(0, 2, 3, 1)
# valid_y = dataset.validation.labels
# test_x = dataset.test.images
# test_x = test_x.reshape([-1, 1, 28, 28]).transpose(0, 2, 3, 1)
# test_y = dataset.test.labels
# valid_x -= train_mean
# test_x -= train_mean


def build_model(inputs, labels, num_classes):
    weight_decay = 1e1
    conv1sz = 16
    pool1sz = 2
    pool1str = 2
    conv2sz = 32
    pool2sz = 2
    pool2str = 2
    fc3sz = 512
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):

        net = layers.convolution2d(inputs, conv1sz, scope='conv1')
        net = layers.max_pool2d(net, pool1sz, stride=pool1str, scope='pool1')
        net = layers.convolution2d(net, conv2sz, scope='conv2')
        net = layers.max_pool2d(net, pool2sz, stride=pool2str, scope='pool2')
#         net = layers.flatten(net, scope='flatten3')
#         net = layers.fully_connected(net, fc3sz, activation_fn=tf.nn.relu,
#                                      weights_regularizer=layers.l2_regularizer(
#                                          weight_decay),
#                                      scope='fc3')

    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=layers.variance_scaling_initializer(),
                                        weights_regularizer=layers.l2_regularizer(weight_decay)):

        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(net)
        net = layers.fully_connected(net, fc3sz, scope='fc3')

    logits = layers.fully_connected(
        net, num_classes, activation_fn=None, scope='logits')
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return logits, loss

inputs = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputs')
labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
net, loss = build_model(inputs, labels, 10)
session = tf.InteractiveSession()
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
session.run(tf.initialize_all_variables())
train_size = train_x.shape[0]
batch_size = train_size // 1000
n_epochs = 8
for epoch in range(1, n_epochs + 1):
    indices = np.random.permutation(train_size)
    tr_x = train_x[indices]
    tr_y = train_y[indices]
    for i in range(train_size // batch_size):
        x = tr_x[batch_size * i: batch_size * (i + 1), :]
        y_ = tr_y[batch_size * i: batch_size * (i + 1), :]        
        if i % 100 == 0:
            loss_val = loss.eval(feed_dict={inputs: x, labels: y_})
            print('epoch: {}, loss: {}'.format(epoch, loss_val))
            var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
            val = var.eval()
            nn.draw_conv_filters(epoch, i, attrdict(
                C=1, weights=val.transpose().reshape(16, -1), name='weights'),
                SAVE_DIR)
        train_step.run(feed_dict={inputs: x, labels: y_})
