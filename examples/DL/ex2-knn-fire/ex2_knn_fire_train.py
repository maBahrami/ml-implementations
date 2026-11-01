import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

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

# spliting the dataset into test and train
x_train, x_test, y_train, y_test = train_test_split(feature_vecotrs, labels, test_size=0.2)

# selecting the classifier method
clf = KNeighborsClassifier(n_neighbors=5)

# fitting in KNN actually means saving all the training data points
clf.fit(x_train, y_train)

# predicting the output of the x_train and calculating the accuracy with respect to the test labels (y_test)
accuracy = clf.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# saving the model
dump(clf, "model/KNN_classifier_fire.joblib")



