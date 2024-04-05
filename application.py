from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from src.vehicle.pipeline.predict_pipeline import CustomData,PredictPipeline
from flask import Flask, render_template, request
import numpy as np
from src.vehicle.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Fuel_Type_Petrol = 1 if request.form['Fuel_Type'] == 'Petrol' else 0
        Seller_Type_Individual = 1 if request.form['Seller_Type'] == 'Individual' else 0
        Transmission_Mannual = 1 if request.form['Transmission'] == 'Mannual' else 0

        custom_data = CustomData(
            Year=Year,
            Selling_Price=0,  # Dummy value
            Present_Price=Present_Price,
            Kms_Driven=Kms_Driven,
            Fuel_Type='Petrol',  # Dummy value
            Seller_Type='Individual',  # Dummy value
            Transmission='Mannual',  # Dummy value
            Owner=Owner
        )
        features_df = custom_data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(features_df.values)

        if results < 0:
            return render_template('home.html', prediction_text="Sorry you cannot sell this car")
        else:
            return render_template('home.html', prediction_text="You Can Sell The Car at {}".format(results))

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)