import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("data/lenna.png", 0) # 0 makes it gray

# kernels
kernel_vertical_1 = np.array([[-1, 1]])
kernel_vertical_2 = np.array([[1, -1]])
kernel_vertical_1_2 = np.array([[1, -1, -1, 1]])

kernel_horizontal_1 = np.array([[-1],
                                [1]])
kernel_horizontal_2 = np.array([[1],
                                [-1]])
kernel_horizontal_1_2 = np.array([[-1],
                                [1],
                                [1],
                                [-1]])

# computing convolution usintg cv2.filter2D
img_conv_vertical_1 = cv2.filter2D(img, cv2.CV_8U, kernel_vertical_1)
img_conv_vertical_2 = cv2.filter2D(img, cv2.CV_8U, kernel_vertical_2)
img_conv_vertical_1_2 = cv2.filter2D(img, cv2.CV_8U, kernel_vertical_1_2)

img_conv_horizontal_1 = cv2.filter2D(img, cv2.CV_8U, kernel_horizontal_1)
img_conv_horizontal_2 = cv2.filter2D(img, cv2.CV_8U, kernel_horizontal_2)
img_conv_horizontal_1_2 = cv2.filter2D(img, cv2.CV_8U, kernel_horizontal_1_2)

# showing the results
images = [img, img_conv_vertical_1, img_conv_vertical_2, img_conv_vertical_1_2,
          img, img_conv_horizontal_1, img_conv_horizontal_2, img_conv_horizontal_1_2]
titles = ["original image", "Vertical 1: [-1 1]", "Vertical 2: [1 -1]", "Vertical 1&2: [1 -1 -1 1]",
          "original image", "Horizontal 1: [-1 1]$^T$", "Horizontal 2: [1 -1]$^T$", "Horizontal 1&2: [-1 1 1 -1]$^T$"]

fig, axes = plt.subplots(2, 4, figsize=(12, 7))
for ax, im, t in zip(axes.flat, images, titles):
    ax.imshow(im, cmap="gray")
    ax.set_title(t)
    ax.axis("off")
plt.tight_layout()
plt.show()