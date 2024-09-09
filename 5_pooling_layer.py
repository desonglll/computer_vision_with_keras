import cv2
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define Image Size
width, height, channels = (224, 224, 3)

# Set seed
tf.random.set_seed(0)

# Load the image
img = cv2.imread("lena/lena_std.jpg")
img = cv2.resize(img, (width, height))

# CNN model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(width, height, channels), filters=64, kernel_size=(3, 3)))

feature_map = model.predict(np.array([img]))

feature_img = feature_map[0, :, :, 0]
print(feature_img)
plt.imshow(feature_img, cmap="gray")
plt.show()

# cv2.imshow("Lena", img)
# cv2.waitKey(0)
