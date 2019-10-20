import numpy as np
import cv2

CATEGORIES = ["Strawberry","Mango"]



def prepare(test1):

    new_array = cv2.resize(test1, (100, 100))
    return new_array.reshape(-1, 100, 100, 1)


import tensorflow as tf

model = tf.keras.models.load_model("64x3-CNN.model")
test=cv2.imread('jyo.jpg',0)
test=test.astype(float)


prediction = model.predict([prepare(test)])
print(prediction)

print(CATEGORIES[int(prediction[0][0])])

