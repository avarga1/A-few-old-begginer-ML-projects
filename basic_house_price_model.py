#!/usr/bin/env python
# coding: utf-8

'''
The following code is an extremely basic example of how to use the model to make predictions.
we define a variable called house_model and assign it the value of the model we just created.
wherein xs is the number of bedrooms and ys is the price of the house in 1000_000s.
'''

import tensorflow as tf
import numpy as np



def house_model():
    # this represents the number of bedrooms
    xs = np.array([1,2,3,4,5,6], dtype=float)
    # this represents the price of the house in 1000_000s
    ys = np.array([1,1.5,2,2.5,3,3.5], dtype=float)
    # create a sequential model with one layer
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    # compile the model with the optimizer and loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')
    # 'train' the model on the data we defined above, over 1000 epochs
    model.fit(xs, ys, epochs=1000)

    # return the model
    return model

# assign the model to a variable
model = house_model()


# predict the price of a house with 7 bedrooms
new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)

