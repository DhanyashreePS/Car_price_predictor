import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("car_data.csv")

# Select only needed columns
data = data[['Year','Present_Price','Kms_Driven','Owner','Selling_Price']]

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("car_model.pkl","wb"))

print("Model saved successfully!")