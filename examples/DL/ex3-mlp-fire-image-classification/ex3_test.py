from tensorflow.keras import models
from joblib import load
import glob
import os
import cv2
import numpy as np


net = models.load_model("model/mlp_classifier_fire.keras")

labels = ["fire", "not fire"]

for address in glob.glob("data/fire_test/*"):

    # Skip folders
    if os.path.isdir(address):
        continue

    img = cv2.imread(address)

    # Skip non-image files (cv2.imread returns None for invalid formats)
    if img is None:
        continue

    img_prep = cv2.resize(img, (32, 32))
    img_prep = img_prep / 255.0
    img_prep = img_prep.flatten()
    # print(img_prep.shape)

    y_pred = net.predict(img_prep.reshape(1, -1))
    prediction = labels[np.argmax(y_pred[0])]
    prob = np.max(y_pred[0])*100
    print(prediction)

    text = f"{prediction}: {prob:.2f}%" # the text to be shown on the image

    # the color of the text depending on the predicted category, either red or green
    if prediction == "fire":
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    # printing the text on the image and showing the image
    cv2.putText(img, str(text), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.95, color, 1)
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord("n"): continue
    cv2.destroyAllWindows()

