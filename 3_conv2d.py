from math import sqrt

import cv2
import keras
from keras import layers
import numpy as np

# Load and preprocess image
img = cv2.imread("lena/lena_std.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))

height, width = img.shape
print(f"Height: {height}, Width: {width}")

# cv2.imshow("Lena", img)
# cv2.waitKey(0)

filter_size = 64
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(height, width, 1), filters=filter_size, kernel_size=(7, 7)))

model.summary()

# Access layers parameters
filters, _ = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()

# Normalized
filters = (filters - f_min) / (f_max - f_min)

print(f"f_min: {f_min}, f_max: {f_max}")

print(f"Filters Shape: {filters.shape}")

# Display filters
for i in range(filter_size):
    f = filters[:, :, :, i]
    # It has 9 pixels and each one is slightly different of others ⬇️
    f = cv2.resize(f, (250, 250), interpolation=cv2.INTER_NEAREST)
    # print(f"f: {f}")
    # cv2.imshow(str(i), f)

cv2.waitKey(0)
