# -*- coding : utf-8 -*-
# @Time      : 2020/11/24 20:49
# @Author    : A WAYS AWAY
# @File      : GradientDescentDemo.py
# @IDE       : PyCharm

import matplotlib.pyplot as plt
import numpy as np
import csv

if __name__ == '__main__':
    x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
    y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

    x = np.arange(-200, -100, 1)  # bias
    y = np.arange(-5, 5, 0.1)  # weight
    Z = np.zeros((len(x), len(y)))
    X, Y = np.meshgrid(x, y)  # 网格点坐标矩阵
    for i in range(len(x)):
        for j in range(len(y)):
            b = x[i]
            w = y[j]
            Z[j][i] = 0
            for n in range(len(x_data)):
                Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
            Z[j][i] = Z[j][i] / len(x_data)

    # ydata = b + w * xdata
    b = -120  # initial b
    w = -4  # initial w
    lr = 1  # learning rate
    iteration = 100000

    # different parameters different learning rates
    b_lr = 0.0
    w_lr = 0.0

    # store initial values for plotting
    b_history = []
    w_history = []

    # Iterations
    for i in range(iteration):
        b_grad = 0.0
        w_grad = 0.0
        for n in range(len(x_data)):
            b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
            w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

        # learning rate
        b_lr = b_lr + b_grad ** 2
        w_lr = w_lr + w_grad ** 2

        # update parameters
        b = b - lr / np.sqrt(b_lr) * b_grad
        w = w - lr / np.sqrt(w_lr) * w_grad

        # store parameters for plotting
        b_history.append(b)
        w_history.append(w)

    print(min(b_history))  # -188.3668387495323
    print(max(w_history))  # 2.6692640713379903

    # plot the figure
    plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
    plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')  # 已知[-188.4] [2.67]
    plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
    plt.xlim(-200, -100)
    plt.ylim(-5, 5)
    plt.xlabel(r'$b$', fontsize=16)
    plt.ylabel(r'$w$', fontsize=16)
    plt.savefig('tuning the learning rate')
    plt.show()


