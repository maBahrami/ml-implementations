from keras import models
import glob
import os
import cv2
import numpy as np


net = models.load_model("model/cnn_fire.keras")

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
    # print(img_prep.shape)
    img_prep = np.expand_dims(img_prep, axis=0)      # (1, 32, 32, 3)
    # Keras models always work on batches of inputs, not on single samples. 
    # The first dimension of the input is the batch size.
    # A single image has shape (32, 32, 3). Without a batch dimension, 
    # Keras would treat the first 32 as the batch size and the remaining (32, 3) 
    # as the image, which doesn’t match what the model expects. 
    # expand_dims(..., axis=0) adds a leading axis and turns it into (1, 32, 32, 3), 
    # a batch that holds one image.
    # The output follows the same pattern. 
    # y_pred comes back with shape (1, 2), one row per image in the batch. 
    # That’s why the code reads y_pred[0] to get the probabilities for your single image.

    y_pred = net.predict(img_prep, verbose=0)
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

