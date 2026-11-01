from joblib import load
import glob
import os
import cv2

model = load("model/KNN_classifier_fire.joblib")

for address in glob.glob("data/*"):

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

    y_pred = model.predict(img_prep.reshape(1, -1))
    prediction = y_pred[0]
    print(prediction)

    cv2.putText(img, prediction, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.95, (0, 255, 100), 2)
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord("n"): continue
    cv2.destroyAllWindows()

