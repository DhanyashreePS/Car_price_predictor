from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("car_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html",prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form['year'])
    price = float(request.form['price'])
    kms = int(request.form['kms'])
    owner = int(request.form['owner'])

    data = np.array([[year, price, kms, owner]])

    prediction = model.predict(data)

    return render_template("index.html",prediction_text="Predicted Price: {} lakhs".format(round(prediction[0],2)))

if __name__ == "__main__":
    app.run(debug=True)