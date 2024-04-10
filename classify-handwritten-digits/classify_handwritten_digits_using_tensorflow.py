# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:24:04 2024

@author: Reza
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first image from the training dataset
plt.figure()
plt.imshow(train_images[0], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()



# Assign data and labels
data = train_images
labels = train_labels
test_data = test_images
test_labels = test_labels

def display(i):
	img = test_data[i]
	plt.title('label : {}'.format(test_labels[i]))
	plt.imshow(img.reshape((28, 28)))
	
# image in TensorFlow is 28 by 28 px
display(0)



