import numpy as np
import matplotlib.pyplot as plt
import math


class Random2DGaussian:
    def __init__(self, min_x: int = 0, max_x: int = 10, min_y: int = 0, max_y: int = 10):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        mu_x = (max_x - min_x) * np.random.random_sample() + min_x
        mu_y = (max_y - min_y) * np.random.random_sample() + min_y
        self.mu = np.array([mu_x, mu_y])
        eigen_val_x = ((max_x - min_x) / 5) ** 2 * np.random.random_sample()
        eigen_val_y = ((max_y - min_y) / 5) ** 2 * np.random.random_sample()
        d = np.array([[eigen_val_x, 0], [0, eigen_val_y]])
        alpha = np.random.random_sample() * math.pi
        r = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
        self.sigma = r.transpose() * d * r

    def get_sample(self, n: int):
        return np.array([np.random.multivariate_normal(self.mu, self.sigma) for _ in range(n)])


def sample_gmm_2d(k, c, n):
    distributions = np.concatenate([Random2DGaussian().get_sample(n) for _ in range(k)])
    labels = np.concatenate([np.array([np.random.randint(c)] * n) for _ in range(k)])
    return distributions, labels


def sample_gauss_2d(c, n):
    return np.vstack(Random2DGaussian().get_sample(n) for _ in range(c)), \
           np.concatenate([[i] * n for i in range(c)])


def graph_data(x, y_, y):
    tp = x[(y_ == 1) & (y == 1)]
    tn = x[(y_ == 0) & (y == 0)]
    fp = x[(y_ == 0) & (y == 1)]
    fn = x[(y_ == 1) & (y == 0)]
    # plt.scatter(x[:, 0], x[:, 1])
    plt.scatter(tp[:, 0], tp[:, 1], color='r', marker='o', edgecolor='k')
    plt.scatter(tn[:, 0], tn[:, 1], color='b', marker='o', edgecolor='k')
    plt.scatter(fp[:, 0], fp[:, 1], color='b', marker='s', edgecolor='k')
    plt.scatter(fn[:, 0], fn[:, 1], color='r', marker='s', edgecolor='k')


def graph_surface(fun, rect, offset=0., atol=1e-3):
    ([x_min, y_min], [x_max, y_max]) = rect
    dx = x_max - x_min
    dy = y_max - y_min
    x_min -= dx / 10
    y_min -= dy / 10
    x_max += dx / 10
    y_max += dy / 10
    x_line = np.linspace(x_min, x_max, 400)
    y_line = np.linspace(y_min, y_max, 400)
    mesh_grid = np.meshgrid(x_line, y_line)
    mesh = np.array(fun(np.dstack(mesh_grid).reshape(-1, 2)) - offset).reshape(len(x_line), -1)
    xs, ys = mesh_grid
    v_min = np.min(mesh)
    v_max = np.max(mesh)
    plt.pcolormesh(xs, ys, mesh, vmin=v_min, vmax=v_max)
    plt.rcParams['contour.negative_linestyle'] = 'dashed'
    plt.contour(xs, ys, mesh, levels=[0.], colors='k')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


def my_dummy_decision(x):
    scores = x[:, 0] + x[:, 1] - 5
    return scores > 0.5
