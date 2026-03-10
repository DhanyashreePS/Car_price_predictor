from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("car_model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html", prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():

    company = request.form['company']
    year = int(request.form['year'])
    kms = int(request.form['kms'])
    fuel = request.form['fuel']

    # Convert to DataFrame (IMPORTANT)
    data = pd.DataFrame({
        'company':[company],
        'year':[year],
        'kms_driven':[kms],
        'fuel_type':[fuel]
    })

    prediction = model.predict(data)

    price = round(prediction[0]/100000,2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: {price} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)