import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import lab1.fcann2 as fcann2
import lab1.data as data
from lab1.tflogreg import TFLogReg
from lab1.tfdeep import TFDeep


def demo_fcann2():
    np.random.seed(900)
    x, y_ = data.sample_gmm_2d(k=8, c=2, n=10)
    parameters = fcann2.fcann2_trainer(x, y_)
    y = fcann2.fcann2_classifier(x, parameters)[:, 1] > 0.5
    rect = (np.min(x, axis=0), np.max(x, axis=0))
    data.graph_surface(lambda _x: fcann2.fcann2_classifier(_x, parameters)[:, 1], rect, offset=0.5)
    data.graph_data(x, y_, y)
    plt.show()


def demo_tflogreg():
    np.random.seed(100)
    tf.set_random_seed(100)
    x, y_ = data.sample_gmm_2d(k=6, c=2, n=20)
    c = max(y_) + 1
    y_one_hot_ = np.zeros((len(y_), c))
    y_one_hot_[np.arange(len(y_)), y_] = 1

    tflr = TFLogReg(x.shape[1], y_one_hot_.shape[1], 0.005)
    tflr.train(x, y_one_hot_, 10000)
    probs = tflr.eval(x)
    y = probs[:, 1] > 0.5
    rect = (np.min(x, axis=0), np.max(x, axis=0))
    data.graph_surface(lambda _x: tflr.eval(_x, )[:, 1], rect, offset=0.5)
    data.graph_data(x, y_, y)
    plt.show()
    accuracy = np.sum(y * y_ + (1 - y) * (1 - y_)) / len(y_)
    precision = np.sum(y * y_ / np.sum(y * y_ + y * (1 - y_)))
    recall = np.sum(y * y_ / np.sum(y * y_ + (1 - y) * y_))
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))


def demo_tfdeep():
    np.random.seed(100)
    tf.set_random_seed(100)
    x, y_ = data.sample_gmm_2d(k=6, c=2, n=100)
    c = max(y_) + 1
    y_one_hot_ = np.zeros((len(y_), c))
    y_one_hot_[np.arange(len(y_)), y_] = 1

    tflr = TFDeep([x.shape[1], 10, 10, y_one_hot_.shape[1]], param_delta=5e-3, param_lambda=1e-2)
    tflr.train(x, y_one_hot_, 10000)
    probs = tflr.eval(x)
    y = probs[:, 1] > 0.5
    rect = (np.min(x, axis=0), np.max(x, axis=0))
    data.graph_surface(lambda _x: tflr.eval(_x)[:, 1], rect, offset=0.5)
    data.graph_data(x, y_, y)
    plt.show()
    accuracy = np.sum(y * y_ + (1 - y) * (1 - y_)) / len(y_)
    precision = np.sum(y * y_ / np.sum(y * y_ + y * (1 - y_)))
    recall = np.sum(y * y_ / np.sum(y * y_ + (1 - y) * y_))
    print('accuracy: {}'.format(accuracy))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))


def count_parameters():
    for variable in tf.trainable_variables():
        print(variable.name, np.product(variable.get_shape()))
    print('total parameters:', np.sum(list(map(lambda _x: np.product(_x.get_shape()), tf.trainable_variables()))))


if __name__ == "__main__":
    demo_tflogreg()
    count_parameters()
