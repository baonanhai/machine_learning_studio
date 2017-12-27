#! /usr/bin/python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10) -> None:
        super().__init__()
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # 生成初始化权重以及阈值
        self.errors_ = []  # 错误计数
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))  # 计算更新量
                self.w_[1:] += update * xi  # 更新权重
                self.w_[0] += update  # 更新阈值
                errors += int(update != 0.0)  # 更新错误计数
            self.errors_.append(errors)  # 错误计数变更历史
        print('错误计数历史:', self.errors_)
        return self

    def net_input(self, X):
        """计算样本所有属性和权重相乘的和（两个向量点积）以及和阈值做处理"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """激活函数计算新标签，单位阶跃函数"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 生成颜色样板

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 取样本一个属性的最大最小值
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 取样本另外一个属性的最大最小值
    # 生成点格数据
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # 把点格数据拿出来做分类
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


def show_source_data(X):
    """绘制原始数据"""
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()


def show_errors(errors):
    """绘制训练过程中的错误计数（越来越小就是收敛了）"""
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassification')
    plt.show()


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    # show_source_data(X)
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    # show_errors(ppn.errors_)
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [xm]')
    plt.ylabel('pepal length [xm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
