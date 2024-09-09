import keras
import cv2
from keras import layers
import numpy as np

img_size = 32
channels = 3
img = cv2.imread("lena/lena_std.jpg")
img = cv2.resize(img, (img_size, img_size))

# Model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(img_size, img_size, channels), filters=64, kernel_size=(3, 3)))
model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(units=10))
model.summary()
result = model.predict(np.array([img]))
print(result.shape)
print(result)
