import numpy as np
import common as cm
import time
import datetime
import random


class RamdonClassifier(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.ytr = Y

    def predict(self, X, Yte):
        cnt = 0
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(1000):
            t = time.time()
            time_begin = int(round(t * 1000))
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distance)
            Ypred[i] = self.ytr[0, min_index]
            t = time.time()
            time_end = int(round(t * 1000))
            print("data-%05d: process-time-%.6fs" %
                  (i, (time_end - time_begin)/1000), "-", Ypred[i] == Yte[0, i])
            if Ypred[i] == Yte[0, i]:
                cnt += 1
        return cnt
        # return Ypred


t = time.time()
time_begin = int(round(t * 1000))
maxn = 10000
nn = RamdonClassifier()
Xtr, Ytr, Xte, Yte = cm.load_CIFAR10()
nn.train(Xtr, Ytr)
cnt = 0
# Pte = nn.predict(Xte, Yte)
# for i in range(maxn):
#     if Pte[i] == Yte[0, i]:
#         cnt += 1
cnt = nn.predict(Xte, Yte)
t = time.time()
time_end = int(round(t * 1000))
print("total-process-time-%.6fs" % ((time_end - time_begin)/1000))
print("accuracy = %.2f%%" % (cnt/maxn*100))
