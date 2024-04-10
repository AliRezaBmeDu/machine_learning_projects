# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:24:04 2024

@author: Reza
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.run_functions_eagerly(True)
# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten images
train_images = train_images.reshape(-1, 28 * 28)
test_images = test_images.reshape(-1, 28 * 28)

# Define a function for preprocessing
def preprocess(image, label):
    return image / 255.0, label

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Apply preprocessing using map function
train_dataset, train_labels = train_dataset.map(preprocess)
test_dataset. test_labels = test_dataset.map(preprocess)

# Define a simple linear model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28 * 28,)),  # Flatten the input images
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 units and softmax activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, train_labels, epochs=5, batch_size=100)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_dataset, test_labels)
print('Test accuracy:', test_accuracy)
