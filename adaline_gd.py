#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd


class AdalineGd(object):
    def __init__(self, eta=0.01, n_iter=10) -> None:
        super().__init__()
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # 生成初始化权重以及阈值
        self.cost_ = []  # 错误计数
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """计算样本所有属性和权重相乘的和（两个向量点积）以及和阈值做处理"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        """激活函数计算新标签，单位阶跃函数"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def main():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('petal length')
    # plt.ylabel('sepal length')
    # plt.legend(loc='upper left')
    # plt.show()
    ppn = AdalineGd(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassification')
    # plt.show()


if __name__ == '__main__':
    main()
