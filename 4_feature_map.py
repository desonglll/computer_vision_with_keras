import keras
import cv2
from keras import layers
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lena/lena_std.jpg")
img = cv2.resize(img, (224, 224))

b, g, r = cv2.split(img)
cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.imshow("Lena", img)
cv2.waitKey(0)

# Model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3)))

model.summary()

feature_map = model.predict(np.array([img]))

print(feature_map.shape)

# Display the feature map
for i in range(64):
    feature_img = feature_map[0, :, :, i]
    ax = plt.subplot(8, 8, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")

plt.show()
