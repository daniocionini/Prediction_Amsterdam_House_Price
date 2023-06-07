# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

np.random.seed(42) # set random seed


# -------------------------------------- Data Import
data = pd.read_csv('./regression_df.csv')



# -------------------------------------- Model Target
TARGET = "Price" 


# -------------------------------------- Train and Test CSV
data_train = pd.read_csv("./data/data_train.csv")
data_test = pd.read_csv("./data/data_test.csv")
