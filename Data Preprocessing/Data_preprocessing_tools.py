#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_scv('Data_csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x[:, 1:3])
x[: , 1:3] = imputer.transform(x[:, 1:3])

#Encoding the independent variable
from sklearn.compose import ColumnTransformer
