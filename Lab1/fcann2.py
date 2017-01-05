import numpy as np


def fcann2_trainer(x, y_, param_niter=1e4, param_delta=1e-3, param_lambda=1e-5, hidden_layer_dim=5, parameters=None):
    n, d = x.shape
    c = np.max(y_) + 1
    y_one_hot = np.zeros((len(y_), c))
    y_one_hot[np.arange(len(y_)), y_] = 1
    if parameters is not None:
        w1, b1, w2, b2 = parameters
    else:
        w1 = np.random.randn(hidden_layer_dim, c)
        b1 = np.random.randn(hidden_layer_dim, 1)
        w2 = np.random.randn(c, hidden_layer_dim)
        b2 = np.random.randn(c, 1)
    for it in range(int(param_niter) + 1):
        p = fcann2_classifier(x, (w1, b1, w2, b2))
        loss = -np.mean(np.log(np.sum(p * y_one_hot, axis=1)))
        if it % 1000 == 0:
            print("iteration: #{}, loss: {}".format(it, loss))
        s1, h1, s2 = calculate_layers(w1, b1, w2, b2, x)
        diff_p_y_ = p - y_one_hot
        dw2 = np.dot(diff_p_y_.transpose(), h1) / n
        db2 = np.mean(diff_p_y_, axis=0).reshape(-1, 1)
        w2 -= param_delta * dw2 - param_lambda * w2
        b2 -= param_delta * db2
        diff_p_y__w2 = np.dot(diff_p_y_, w2)
        diag_s1 = np.array([np.diag(s_ > 0) for s_ in s1])
        diff_p_y__w2_diag_s1 = np.einsum('ij,ijk->ij', diff_p_y__w2, diag_s1)
        dw1 = np.dot(diff_p_y__w2_diag_s1.transpose(), x) / n
        db1 = np.mean(diff_p_y__w2_diag_s1, axis=0).reshape(-1, 1)
        w1 -= param_delta * dw1 - param_lambda * w1
        b1 -= param_delta * db1
    return w1, b1, w2, b2


def fcann2_classifier(x, parameters):
    w1, b1, w2, b2 = parameters
    s1, h1, s2 = calculate_layers(w1, b1, w2, b2, x)
    return np.apply_along_axis(soft_max, 1, s2)


def calculate_layers(w1, b1, w2, b2, x):
    s1 = np.dot(np.array([x]), w1.transpose())[0] + b1.transpose()
    h1 = np.maximum(0, s1)
    s2 = np.dot(np.array([h1]), w2.transpose())[0] + b2.transpose()
    return s1, h1, s2


def fcann2_single_classifier(x, parameters):
    w1, b1, w2, b2 = parameters
    s1 = np.dot(w1.transpose(), x) + b1
    h1 = np.maximum(0, s1)
    s2 = np.dot(w2.transpose(), h1) + b2
    return soft_max(s2)


def divide_large(a, b):
    if np.isinf(b):
        if np.isinf(a):
            return 1.
        else:
            return 0.
    else:
        return a / b


def soft_max(s):
    exp_s = np.exp(s - np.max(s))
    return exp_s / np.sum(exp_s)
