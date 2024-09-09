import cv2
import keras
from keras import layers
import numpy as np

# BGR

# Load and preprocess image
img = cv2.imread("lena/lena_std.jpg")
height, width, channels = img.shape
print(f"Height: {height}, Width: {width}, Channels: {channels}")

# Create a Sequential model
model = keras.Sequential()
model.add(layers.Input(shape=(height, width, channels)))
model.add(layers.Dense(32))
model.add(layers.Dense(16))
model.add(layers.Dense(2))

preprocessed_img = np.array([img])
print(f"Preprocessed Image: {preprocessed_img.shape}")

result = model(preprocessed_img)

model.summary()
