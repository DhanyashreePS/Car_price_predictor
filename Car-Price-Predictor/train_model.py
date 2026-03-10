import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("quikr_car.csv")

# Remove missing values
data = data.dropna()

# Convert year
data['year'] = data['year'].astype(str)
data = data[data['year'].str.isnumeric()]
data['year'] = data['year'].astype(int)

# Clean Price column
data = data[data['Price'] != 'Ask For Price']
data['Price'] = data['Price'].astype(str).str.replace(',', '')
data = data[data['Price'].str.isnumeric()]
data['Price'] = data['Price'].astype(int)

# Clean kms_driven column
data['kms_driven'] = data['kms_driven'].astype(str).str.replace(',', '')
data['kms_driven'] = data['kms_driven'].str.replace(' kms', '')
data = data[data['kms_driven'].str.isnumeric()]
data['kms_driven'] = data['kms_driven'].astype(int)

# Features and target
X = data[['company','year','kms_driven','fuel_type']]
y = data['Price']

# Encode categorical columns
ohe = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    [('encoder', ohe, ['company','fuel_type'])],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline([
    ('transform', column_trans),
    ('lr', LinearRegression())
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("car_model.pkl", "wb"))

print("Model trained and saved successfully!")