# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:33:51 2024

@author: Reza
"""

from datetime import datetime 
import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
import numpy as np 
import seaborn as sns 


microsoft = pd.read_csv('../datasets/MicrosoftStock.csv') 
print(microsoft.head()) 



