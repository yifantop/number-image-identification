import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    exp_a_sum = np.sum(exp_a)

    return exp_a / exp_a_sum


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    delta = 1e-7
    batch_size = y.shape[0]

    return -np.sum(t * np.log(y + delta)) / batch_size


def numerical_gradient(f, W_name, W, net):
    h = 1e-4
    grad = np.zeros_like(W)

    it = np.nditer(W, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        tmp_val = W[it.multi_index]
        net.params[W_name][it.multi_index] = tmp_val + h
        fxh1 = f(W)
        net.params[W_name][it.multi_index] = tmp_val - h
        fxh2 = f(W)
        net.params[W_name][it.multi_index] = tmp_val

        grad[it.multi_index] = (fxh1 - fxh2) / (2 * h)

        it.iternext()

    return grad
