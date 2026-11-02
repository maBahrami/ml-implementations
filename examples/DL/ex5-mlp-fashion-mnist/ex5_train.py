import cv2
import numpy as np
import pandas as pd
from keras import models, layers
import matplotlib.pyplot as plt
import os


# reading the train data and label
x_train = []
y_train = []

print("------- reading training images started ---------")
df = pd.read_csv("data/fashion_mnist/train.csv")

for i, row in df.iterrows():

    img_address = img_address = f"data/fashion_mnist/train/{row['Unnamed: 0']}.png"
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("not found:", img_address)
    img = img.flatten() / 255.0
    x_train.append(img)

    label = row['Class']
    y_train.append(label)

    if i % 10000 == 0:
        print(f"[INFO]  {i} images added")

x_train = np.array(x_train)
y_train = np.array(y_train)
# print(np.array(x_train).shape)
# print(len(y_train))


print("------- reading validation images started --------")
# reading the validation data and label
x_val = []
y_val = []

df = pd.read_csv("data/fashion_mnist/val.csv")

for i, row in df.iterrows():

    img_address = f"data/fashion_mnist/val/{row['Unnamed: 0']}.png"
    img = cv2.imread(img_address, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("not found:", img_address)
        continue
    img = img.flatten() / 255.0
    x_val.append(img)

    label = row['Class']
    y_val.append(label)

    if i % 2000 == 0:
        print(f"[INFO]  {i} images added")

x_val = np.array(x_val)
y_val = np.array(y_val)
# print(np.array(x_val).shape)
# print(len(y_val))


# checking some of the labels to make sure that the labels are read correctly
# print(x_train.shape, x_train.min(), x_train.max(), x_train.mean())   # expect max 1.0, mean ~0.29
# print(np.unique(y_train, return_counts=True))                         # expect 0–9, ~6000 each

# fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# for ax, i in zip(axes.flat, range(10)):
#     ax.imshow(x_train[i].reshape(28, 28), cmap="gray")
#     ax.set_title(y_train[i])
#     ax.axis("off")
# plt.show()


# model configuration
net = models.Sequential([
                        layers.Input(shape=(784,)),
                        layers.Dense(128, activation="sigmoid"),
                        layers.Dense(10, activation="softmax")   
                        ])

print(net.summary())

net.compile(
            optimizer = "SGD",
            metrics = ["accuracy"],
            loss = "sparse_categorical_crossentropy"
)

H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_val, y_val), epochs=10)

plt.plot(H.history["accuracy"], label = "train accuracy")
plt.plot(H.history["val_accuracy"], label = "validation accuracy")
plt.plot(H.history["loss"], label = "train loss")
plt.plot(H.history["val_loss"], label = "validation loss")
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.title("Fashion MNIST classification")
plt.legend()
plt.show()

