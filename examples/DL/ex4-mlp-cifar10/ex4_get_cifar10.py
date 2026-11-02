# You end up with the same data as cifar10.load_data(): 
# uint8 arrays of shape (50000, 32, 32, 3) and (10000, 32, 32, 3), 
# plus integer labels. The only difference is that the labels are 1-D here, 
# while Keras gives shape (N, 1). sparse_categorical_crossentropy accepts either, 
# so this doesn’t matter for training.

import numpy as np
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")
x_train = np.stack([np.array(im) for im in ds["train"]["img"]])
y_train = np.array(ds["train"]["label"])
x_test  = np.stack([np.array(im) for im in ds["test"]["img"]])
y_test  = np.array(ds["test"]["label"])

np.savez_compressed("data/cifar10.npz", x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)

