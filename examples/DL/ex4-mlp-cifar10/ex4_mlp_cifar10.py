from keras import layers, models
# from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

d = np.load("data/cifar10.npz")
x_train, y_train, x_test, y_test = d["x_train"], d["y_train"], d["x_test"], d["y_test"]

x_train, x_test = x_train/255.0, x_test/255.0

net = models.Sequential([
                        layers.Flatten(),
                        layers.Dense(400, activation = "relu"),
                        layers.Dense(200, activation = "relu"),
                        layers.Dense(80, activation = "relu"),
                        layers.Dense(10, activation = "softmax"),
                        ])

net.compile(
            optimizer = "SGD",
            metrics = ["accuracy"],
            loss = "sparse_categorical_crossentropy"
            )

H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=15)

plt.plot(H.history["accuracy"], label = "train accuracy")
plt.plot(H.history["val_accuracy"], label = "test accuracy")
plt.plot(H.history["loss"], label = "train loss")
plt.plot(H.history["val_loss"], label = "test loss")
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.title("Cifar 10 classification")
plt.legend()
plt.show()
