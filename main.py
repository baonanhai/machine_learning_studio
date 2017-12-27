#! /usr/bin/python3
# -*- coding:utf-8 -*-

import os
import time

import numpy as np

from adaline_gd import AdalineGd
from utils_for_image import get_image_info


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


if __name__ == '__main__':
    time1 = time.time()
    X, y = load_train_data()
    time2 = time.time()
    print('加载数据耗时:', time2 - time1)
    ppn = AdalineGd(eta=0.01, n_iter=100)
    ppn.fit(X, y)
    time3 = time.time()
    print('训练耗时:', time3 - time2)
