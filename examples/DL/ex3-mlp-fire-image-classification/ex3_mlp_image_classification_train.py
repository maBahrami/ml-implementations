import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras import models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

plt.style.use("ggplot")

feature_vecotrs = []
labels = []

# reading the images of the dataset, preprocessing, and constructing the feature vectors and lebels
for i, address in enumerate(glob.glob("data/fire_dataset/*/*")):

    img = cv2.imread(address)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.flatten()

    feature_vecotrs.append(img)
    labels.append(address.split("/")[2])

    if i % 100 == 0:
        print(f"[INFO] {i} images processed")



feature_vecotrs = np.array(feature_vecotrs)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# spliting the dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(feature_vecotrs, labels, test_size=0.2)

net = models.Sequential([
    layers.Input(shape=(3072,)),
    layers.Dense(300, activation="relu"),
    layers.Dense(40, activation="relu"),
    layers.Dense(2, activation="softmax")
])

print(net.summary())

net.compile(optimizer="SGD",
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=20)

loss, acc = net.evaluate(x_test, y_test)
print(f"loss: {loss:.2f}    accuaracy:{acc:.2f}")


plt.plot(H.history["accuracy"], label="train accuracy")
plt.plot(H.history["val_accuracy"], label="test accuracy")
plt.plot(H.history["loss"], label="train accuracy")
plt.plot(H.history["val_loss"], label="test accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Fire Classification")
plt.show()

net.save("model/mlp.keras")
