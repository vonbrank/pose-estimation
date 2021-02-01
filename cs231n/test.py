import common as cm
import numpy as np

cm.debug()
Xtr, Ytr, Xte, Yte = cm.load_CIFAR10()
print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
# Ytr = Ytr.reshape(2, 25000)
print(Ytr)
print(type(Ytr))
# print(Xtr.shape, Ytr.shape)
cm.array_to_image(Xtr)
cm.array_to_image(Xte)
