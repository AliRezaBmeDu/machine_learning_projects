# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 01:24:31 2024

@author: Reza
"""

# importing the hand written digit dataset
from sklearn import datasets

# digit contain the dataset
digits = datasets.load_digits()

# dir function use to display the attributes of the dataset
print(dir(digits))

# outputting the picture value as a series of numbers
print(digits.images[0])


# importing the matplotlib libraries pyplot function
import matplotlib.pyplot as plt
# defining the function plot_multi

def plot_multi(i):
	nplots = 16
	fig = plt.figure(figsize=(15, 15))
	for j in range(nplots):
		plt.subplot(4, 4, j+1)
		plt.imshow(digits.images[i+j], cmap='binary')
		plt.title(digits.target[i+j])
		plt.axis('off')
	# printing the each digits in the dataset.
	plt.show()

plot_multi(0)


