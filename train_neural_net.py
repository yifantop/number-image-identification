from two_layer_net import TwoLayerNet
from mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  flatten=True,
                                                  one_hot_label=True)

train_loss_list = []
iter_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(784, 50, 10)

print("start...")
for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grads = network.numerical_gradient(x_batch, t_batch)

    # 更新权重
    for key in ("W1", "W2", "b1", "b2"):
        network.params[key] -= learning_rate * grads[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    print("epoch " + str(i) + " finished")
    if i % 50 == 0:
        with open("./network.pkl", "wb") as f:
            pickle.dump(network, f)
        with open("./train_loss_record.pkl", "wb") as f:
            pickle.dump(train_loss_list, f)

# 将神经网络存起来
with open("./network.pkl", "wb") as f:
    pickle.dump(network, f)
# 存学习记录
with open("./train_loss_record.pkl", "wb") as f:
    pickle.dump(train_loss_list, f)

print("end")

# fig, ax = plt.subplots()
# ax.set_xlabel("train times")
# ax.set_ylabel("loss value")
# ax.plot(np.arange(0, iter_num, 1), train_loss_list)
# ax.set_title("train loss record")
# plt.show()
