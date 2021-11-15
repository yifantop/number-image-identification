import numpy as np
from numba import jit

from tool import sigmoid, softmax, cross_entropy_error, numerical_gradient


class TwoLayerNet:
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(
            hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    @jit
    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + 2
        y = softmax(a2)

        return y

    @jit
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    @jit
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.sum(y == t) / y.shape[0]
        
    @jit
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, "W1", self.params["W1"], self)
        grads["W2"] = numerical_gradient(loss_W, "W2", self.params["W2"], self)
        grads["b1"] = numerical_gradient(loss_W, "b1", self.params["b1"], self)
        grads["b2"] = numerical_gradient(loss_W, "b2", self.params["b2"], self)

        return grads
