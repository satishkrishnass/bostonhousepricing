import pickle
import numpy as np
import pandas as pd
from flask import Flask, app, jsonify, render_template, request, url_for

app=Flask(__name__)

xgmodel = pickle.load(open('regressionModel.pkl','rb'))
scaler  = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    #data = request.json['data']
    data=[float(x) for x in request.form.values()]
    data_before_scaling = np.array(data).reshape(1,-1)
    data_after_scaling  = scaler.transform(data_before_scaling)
    predicted_output    = xgmodel.predict(data_after_scaling)
    return jsonify(float(predicted_output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=xgmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)