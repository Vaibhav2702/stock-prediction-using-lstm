from flask import Flask, render_template, flash, request
from wtforms import Form, TextAreaField, validators, StringField, SubmitField
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib

import pickle
import numpy as np
import pandas as pd



# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


model = keras.models.load_model("model.h5")
msft = keras.models.load_model("msft.h5")
tsla = keras.models.load_model("TSLA.h5")
fb = keras.models.load_model("FB.h5")
# scaler = joblib.load("scaler.joblib")
scaler= MinMaxScaler(feature_range=(0,1))
print(type(model))
# def index():
#     companies = ['AAPL','MSFT','TSLA','FB']
#     return render_template('index2.html',companies=companies)



def getapidata(company):
    from alpha_vantage.timeseries import TimeSeries
    API_KEY = 'TT49KBDQHV21HOY3'
    ts = TimeSeries(API_KEY,output_format='pandas')
    df1 = ts.get_daily(company,outputsize='full')

    df = pd.DataFrame(df1[0])
    df.sort_index(inplace=True)
    Df = df[['4. close']]
    

    last_60_days = Df[-60:].values
    last_60_days_scaled = scaler.fit_transform(last_60_days)
    x1_test = []
    x1_test.append(last_60_days_scaled)
    x1_test = np.array(x1_test)
    x1_test = np.reshape(x1_test,(x1_test.shape[0],x1_test.shape[1],1))
    # print(x1_test)
    return x1_test









@app.route('/predict', methods=['post', 'get'])

def predict():
    prediction = ''
    if request.method == 'POST':
        com=request.form['company']
        print(request.form)
        test=[]
        for key,item in request.form.items():
            print(key,item)
            test.append(item)
        test2= getapidata(com)
        

  

        print(prediction,'predection')
        if com=='aapl':
            name='APPLE'
            x=model
        elif com=='msft':
            name='MICROSOFT'
            x=msft
        elif com=='tsla':
            name='TESLA'
            x=tsla
        elif com=='fb':
            name='FACEBOOK'
            x=fb
        
        prediction=x.predict(test2)
        st_value=(float)(scaler.inverse_transform(prediction))
        prediction=name+": $"+str(round(st_value,2))
        
            
    return render_template('index2.html', message=prediction)

if __name__ == "__main__":
    app.run()
