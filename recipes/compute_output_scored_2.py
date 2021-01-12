# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import pickle

# Read recipe inputs
models = dataiku.Folder("DQ1im0k6")
models_info = models.get_info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
model_path = models.file_path("finalized_model.sav")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
output = dataiku.Dataset("output")
df = output.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# load the model from disk
loaded_model = pickle.load(open(model_path, 'rb'))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
X_test = df[["bathrooms", "bedrooms", "sqft_living"]].values

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
result = loaded_model.predict(X_test)
df["predicted"] = result


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.


# Write recipe outputs
output_scored_2 = dataiku.Dataset("output_scored_2")
output_scored_2.write_with_schema(df)