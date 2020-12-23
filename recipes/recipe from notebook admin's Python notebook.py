# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from pyts.datasets import load_gunpoint
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train = np.append(X_train, y_train_reshaped, axis=1)
test = np.append(X_test, y_test_reshaped, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
colnames = []
for i in range(0, 151):
    c = "col_"+str(i)
    colnames.append(c)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
train_df.columns = colnames
test_df.columns = colnames

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
py_recipe_output = dataiku.Dataset("ts_train")
py_recipe_output.write_with_schema(train_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
py_recipe_output = dataiku.Dataset("ts_test")
py_recipe_output.write_with_schema(test_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
ts_train = dataiku.Dataset("ts_train")
ts_train.write_with_schema(pandas_dataframe)
ts_test = dataiku.Dataset("ts_test")
ts_test.write_with_schema(pandas_dataframe)