import cv2
import keras
from keras.src.layers import Dense

img = cv2.imread("shameless-wallpaper.jpg", cv2.IMREAD_GRAYSCALE)
print(f"Image Shape: {img.shape}")
height, width = img.shape
print(img)

# Keras model structure
input_layer = keras.Input(shape=(height, width))
print(f"Input Layer: {input_layer.shape}")
Layer_1 = Dense(units=64)(input_layer)
Layer_2 = Dense(units=32)(Layer_1)
output = Dense(2)(Layer_2)

# Define model
model = keras.Model(inputs=input_layer, outputs=output)
model.summary()

cv2.imshow("Shameless", img)
cv2.waitKey(0)
