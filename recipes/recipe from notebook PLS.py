# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from sklearn.model_selection import train_test_split
from custom_PLS.custom_PLS import PLS

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_train = dataiku.Dataset("house_price").get_dataframe()
df_test = dataiku.Dataset("output").get_dataframe()


X_train = df_train[["bathrooms", "bedrooms", "sqft_living"]].values
X_test = df_test[["bathrooms", "bedrooms", "sqft_living"]].values
y_train = df_train["price"].values
y_test = df_test["price"].values


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg = PLS(n_components=2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.fit(X_train, y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
y_predicted = reg.predict(X_test)
y_proba = reg.predict_proba(X_test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.score(X_test, y_test)

df_test["predicted"] = y_predicted
df_test["proba"] = y_proba

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
output_scored = dataiku.Dataset("output_scored")
output_scored.write_with_schema(df_test)