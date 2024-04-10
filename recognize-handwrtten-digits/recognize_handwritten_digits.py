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
