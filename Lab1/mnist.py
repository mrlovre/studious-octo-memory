import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from lab1.tfdeep import TFDeep


def main():
    tf.set_random_seed(100)
    np.random.seed(100)
    tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
    mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
    n = mnist.train.images.shape[0]
    d = mnist.train.images.shape[1]
    c = mnist.train.labels.shape[1]
    x = mnist.train.images
    y_one_hot_ = mnist.train.labels
    tflr = TFDeep([x.shape[1], y_one_hot_.shape[1]], param_delta=1e-3, param_lambda=1e-4, stddev=1e-3)
    tflr.train(x, y_one_hot_, 200)
    w = tflr.params()[0][0]
    for i in range(10):
        plt.figure()
        plt.imshow(w[:, i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == "__main__":
    main()
