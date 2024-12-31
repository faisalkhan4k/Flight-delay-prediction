from flask import *
import joblib
import numpy as np
from datetime import datetime
import data.TestModel
import os

from views import preprocess

user_bp = Blueprint('user_bp', __name__)


@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def admin_home():
    msg = ''
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)



def getParameters():
    parameters = (request.form['addr'])
    print(parameters)

    return parameters

@user_bp.route('/predict',  methods=['POST', 'GET'])
def predict():
    print("hi1")
    parameters = []

    if request.method == 'POST':
        if(preprocess()=="valid"):
            print("hi2")
            parameters = getParameters()
            my_pred = data.TestModel.test_model(parameters)
            print("hi5")
            print(my_pred)
            return render_template('result.html', prediction=my_pred)
        else:
            my_pred=-1
            return render_template('result.html', prediction=my_pred)
    else:
        return render_template('predict.html')

@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")