import numpy as np


PARAM_NITER = 50000
PARAM_DELTA = 1e-2


def binary_logistic_regression_trainer(x, y_):
    n, d = x.shape
    w = np.random.randn(d)
    b = 0.
    for i in range(PARAM_NITER):
        scores = np.dot(w, x.transpose()) + b
        probabilities = np.exp(scores) / (1 + np.exp(scores))
        loss = -np.sum(np.log(probabilities * y_ + (1 - probabilities) * (1 - y_)))
        if i % 1000 == 0:
            print("iteration {}: loss {}, w={}, b={}".format(i, loss, w, b))
        dl_ds = probabilities - y_
        grad_w = np.dot(dl_ds, x)
        grad_b = np.sum(dl_ds)
        w += -PARAM_DELTA * grad_w
        b += -PARAM_DELTA * grad_b
    return w, b


def binary_logistic_regression_classifier(x, w, b):
    s = np.dot(w, x.transpose()) + b
    return np.exp(s) / (1 + np.exp(s))
