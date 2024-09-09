import cv2
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set Seed
tf.random.set_seed(0)

width, height = 224, 224
channels = 3
img = cv2.imread("lena/lena_std.jpg")
img = cv2.resize(img, (width, height))

# VGG16
model = keras.Sequential()

# Block 1
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(width, height, 3)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 2
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(128, kernel_size=(3, 3), padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 3
model.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(256, kernel_size=(3, 3), padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 4
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 5
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.Conv2D(512, kernel_size=(3, 3), padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Top
model.add(layers.Flatten())
model.add(layers.Dense(units=4096, activation="relu"))
model.add(layers.Dense(units=4096, activation="relu"))
# If we want to detect 'cat', 'dog', 'bird'
# the unit of the final Dense layer is 3.
model.add(layers.Dense(units=3, activation="softmax"))

model.build()
model.summary()

# Result
result = model.predict(np.array([img]))
print(result)
