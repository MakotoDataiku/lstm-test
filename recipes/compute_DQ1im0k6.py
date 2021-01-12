# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from sklearn.model_selection import train_test_split
from custom_PLS.custom_PLS import PLS
import pickle

df_train = dataiku.Dataset("house_price").get_dataframe()

X_train = df_train[["bathrooms", "bedrooms", "sqft_living"]].values
y_train = df_train["price"].values

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg = PLS(n_components=2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.fit(X_train, y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
models = dataiku.Folder("DQ1im0k6")
filename = '/finalized_model.sav'
model_path = models.get_path() + filename
pickle.dump(reg, open(model_path, 'wb'))