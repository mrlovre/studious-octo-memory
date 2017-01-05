import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
import skimage as ski
import skimage.io
import tflearn.layers.normalization as normalization


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict_ = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict_

DATA_DIR = '/home/frka/python/Lab2/datasets/CIFAR/'
SAVE_DIR = '/home/frka/python/Lab2/out5/'


def draw_conv_filters(epoch, step, weights, save_dir, cols=8):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = cols
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r + k, c:c + k, :] = w[:, :, :, i]
    filename = 'epoch_%06d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def build_model(inputs, labels, num_classes):
    weight_decay = 1e-2
    conv1sz = 10
    pool1sz = 3
    pool1str = 2
    conv2sz = 32
    pool2sz = 3
    pool2str = 2
    fc3sz = 256
    fc4sz = 128
    with tf.contrib.framework.arg_scope([layers.convolution2d],
                                        padding='VALID', stride=2,
#                                         activation_fn=tf.nn.relu6,
#                                                                                 weights_regularizer=layers.l2_regularizer(
#                                         weight_decay),
                                        weights_initializer=layers.variance_scaling_initializer()):

        net = layers.convolution2d(
            inputs, conv1sz, kernel_size=32, scope='conv1')
        net = normalization.local_response_normalization(net, 2)
        
#         net = layers.avg_pool2d(
#             net, 1, padding='SAME', stride=pool1str, scope='pool1')
#         net = layers.convolution2d(net, conv2sz, kernel_size=3, scope='conv2')
#         net = layers.max_pool2d(
#             net, pool2sz, padding='SAME', stride=pool2str, scope='pool2')
    with tf.contrib.framework.arg_scope([layers.fully_connected],
                                        activation_fn=tf.nn.relu6,
#                                         biases_initializer=layers.init_ops.constant_initializer(
#                                             0.1),
                                        #                                         weights_regularizer=layers.l2_regularizer(
                                        # weight_decay),
                                        weights_initializer=layers.variance_scaling_initializer()):

        # sada definiramo potpuno povezane slojeve
        # ali najprije prebacimo 4D tenzor u matricu
        net = layers.flatten(net)
#         net = layers.fully_connected(net, 10, scope='fc3')
#         net = layers.fully_connected(net, fc4sz, scope='fc4')

#     logits = layers.fully_connected(
#         net, num_classes, activation_fn=None, scope='logits')
    logits = net
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

num_channels = 3
img_width = 32
img_height = 32

train_x = np.ndarray(
    (0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape(
    (-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape(
    (-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

train_x, train_y = shuffle_data(train_x, train_y)
# train_x = train_x[np.logical_and(train_y >= 2, train_y <= 3)]
# train_y = train_y[np.logical_and(train_y >= 2, train_y <= 3)]
# valid_x = train_x[train_size:, ...]
# valid_y = train_y[  train_size:, ...]

train_size = 100
train_x = train_x[:train_size, ...]
train_y = train_y[:train_size, ...]

for i, image in enumerate(train_x):
    ski.io.imsave(SAVE_DIR + 'image%03d.png' % i, np.uint8(image))

data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))
train_x = (train_x - data_mean) / data_std

# valid_x = (valid_x - data_mean) / data_std
# test_x = (test_x - data_mean) / data_std

train_x /= np.max(train_x)

inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
labels = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
loss = build_model(inputs, labels, 10)
session = tf.InteractiveSession()
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
session.run(tf.initialize_all_variables())
batch_size = 10
n_epochs = 5000000
var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
val = var.eval(session)
draw_conv_filters(0, 0, val, SAVE_DIR, cols=5)
prev_loss = 100000
for epoch in range(1, n_epochs + 1):
    indices = np.random.permutation(train_size)
    tr_x = train_x[indices]
    tr_y = train_y[indices]
    tr_y = dense_to_one_hot(tr_y, 10)
    loss_val = 0
    for i in range(train_size // batch_size):
        x = tr_x[batch_size * i: batch_size * (i + 1), :]
        y_ = tr_y[batch_size * i: batch_size * (i + 1)]
        train_step.run(feed_dict={inputs: x, labels: y_})
        var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
        val = var.eval(session)
        loss_val += loss.eval(feed_dict={inputs: x, labels: y_})
    if prev_loss > loss_val:
        prev_loss = loss_val
        draw_conv_filters(epoch, i, val, SAVE_DIR, cols=5)
    print('epoch: {}, loss: {}'.format(epoch, loss_val))
