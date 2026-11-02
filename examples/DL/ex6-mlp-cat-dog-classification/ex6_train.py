import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import SGD

plt.style.use("ggplot")

feature_vector = []
labels = []

for i, address in enumerate(glob.glob("data/PetImages/*/*")):

    img = cv2.imread(address)

    if img is None:
        print("----------------> skipped: ", address)
        continue

    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.flatten()

    feature_vector.append(img)
    labels.append(address.split("/")[2])

    if i % 1000 == 0:
        print(f"[INFO] {i} images processed")


feature_vector = np.array(feature_vector) 

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# print(feature_vector.shape)
# print(len(labels))

# spliting the dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(feature_vector, labels, test_size=0.2)


net = models.Sequential([
    layers.Input(shape=(32*32*3,)),
    layers.Dense(2000, activation="relu"),
    # layers.Dropout(0.5),
    layers.Dense(500, activation="relu"),
    # layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")
                        ])

print(net.summary())

net.compile(optimizer=SGD(learning_rate=0.01),
            loss="categorical_crossentropy",
            metrics=["accuracy"])

H = net.fit(x_train, y_train, batch_size=32, validation_data=(x_test, y_test), epochs=20)

loss, acc = net.evaluate(x_test, y_test)
print(f"loss: {loss:.2f}    accuaracy:{acc:.2f}")


plt.plot(H.history["accuracy"], label="train accuracy")
plt.plot(H.history["val_accuracy"], label="test accuracy")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.title("Cat-Dog Classification")
plt.show()



# Flattening throws away which pixels are next to each other, 
# and that spatial information is what a model needs to tell a cat from a dog.