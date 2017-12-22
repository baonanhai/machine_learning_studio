import os
import time

import numpy as np
import pandas as pd

from utils_for_image import get_image_info


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


def load_train_data():
    X = []
    y = []
    for root, dirs, files in os.walk('train_img', topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_info = os.path.split(file_path)
            X.append(get_image_info(file_path))
            if '1' in file_info[0]:
                flag = 1
            else:
                flag = 0
            y.append(flag)
    return np.array(X), np.array(y),


def test():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    print(y)
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    # plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.xlabel('petal length')
    # plt.ylabel('sepal length')
    # plt.legend(loc='upper left')
    # plt.show()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of misclassification')
    # plt.show()


if __name__ == '__main__':
    time1 = time.time()
    X, y = load_train_data()
    time2 = time.time()
    print('加载数据耗时:', time2 - time1)
    ppn = Perceptron(eta=0.01, n_iter=100)
    ppn.fit(X, y)
    time3 = time.time()
    print('训练耗时:', time3 - time2)
