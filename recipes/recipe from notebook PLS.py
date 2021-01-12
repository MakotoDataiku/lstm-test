# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from sklearn.model_selection import train_test_split
from custom_PLS.custom_PLS import PLS

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
my_dataset = dataiku.Dataset("house_price")
df = my_dataset.get_dataframe()
X = df[["bathrooms", "bedrooms", "sqft_living"]].values
y = df["price"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg = PLS(n_components=2)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.fit(X_train, y_train)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.predict(X_test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.score(X_test, y_test)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
reg.get_params()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
output_scored = dataiku.Dataset("output_scored")
output_scored.write_with_schema(pandas_dataframe)