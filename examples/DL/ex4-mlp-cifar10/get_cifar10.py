import numpy as np
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")
x_train = np.stack([np.array(im) for im in ds["train"]["img"]])
y_train = np.array(ds["train"]["label"])
x_test  = np.stack([np.array(im) for im in ds["test"]["img"]])
y_test  = np.array(ds["test"]["label"])

np.savez_compressed("data/cifar10.npz", x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test)

