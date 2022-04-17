#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST' , 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        pred = "like Pizza"
    elif prediction == 0:
        pred = "don't like Pizza"
    output = pred
    return render_template('index.html', prediction_text='you {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

