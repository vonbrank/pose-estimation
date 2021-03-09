#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a module including some common fuctions for the project '

__author__ = 'Von Brank'

import numpy as np
import pickle
import PIL.Image as Image
import pickle as p
import matplotlib.pyplot as pyplot


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


def load_CIFAR10():
    Xtr = 0
    Ytr = 0
    Xte = 0
    Yte = 0
    for i in range(1, 6):
        data_n = load_file("cifar-10-batches-py/data_batch_" + str(i))
        data_d = np.array(data_n['data'])
        labels_l = np.array(data_n['labels'])
        if i == 1:
            Xtr = data_d
            Ytr = labels_l
        else:
            Xtr = np.concatenate([Xtr, data_d], axis=0)
            Ytr = np.concatenate([Ytr, labels_l], axis=0)
    data_n = load_file("cifar-10-batches-py/test_batch")
    data_d = np.array(data_n['data'])
    labels_l = np.array(data_n['labels'])
    Xte = data_d
    Yte = labels_l
    return Xtr, Ytr.reshape(1, 50000), Xte, Yte.reshape(1, 10000)


def array_to_image(arr):
    rows = arr.shape[0]
    arr = arr.reshape(rows, 3, 32, 32)
    for index in range(10):
        a = arr[index]
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        image = Image.merge("RGB", (r, g, b))
        pyplot.imshow(image)
        pyplot.show()


def debug():
    data_n = load_file("cifar-10-batches-py/data_batch_1")
    print(data_n.keys())

    # array_to_image(data_n['data'])
    # data_d = np.array(data_n['data'])
