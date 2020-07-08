#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:13:35 2020

@author: rahulss
"""

import tensorflow as tf
import numpy as np

from tensorflow import keras

# Basic model 
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 0.0, 3.0, 5.0, 7.0], dtype=float)

# Training the model
model.fit(x,Y, epochs=500)

# Testing the model
mTest = model.predict([10.0])
print(mTest)

# Saving Keras file
keras_file = "linear.h5"
keras.models.save_model(model,keras_file)

# Converting Keras file to tensorflow lite

 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)

#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.post_training_quantize=True
tflite_quantized_model=converter.convert()
open("quantized_model.tflite", "wb").write(tflite_quantized_model)