import numpy as np
import common as cm
import random


class RamdonClassifier(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.Xtr = X
        self.ytr = Y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        # print(Ypred)
        for i in range(num_test):
            Ypred[i] = random.randint(0, 9)
        return Ypred


maxn = 10000
rc = RamdonClassifier()
Xtr, Ytr, Xtr, Yte = cm.load_CIFAR10()
rc.train(Xtr, Ytr)
cnt = 0
Pte = rc.predict(Xtr)
# print(Pte, Yte)
for i in range(maxn):
    if Pte[i] == Yte[0, i]:
        cnt += 1
print("accuracy = %.2f%%" % (cnt/maxn*100))
