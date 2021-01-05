from flask import Flask, request
import pickle
import pandas as pd
import json
import os
import numpy as np

app = Flask(__name__)
MODEL_FILE_NAME = 'tyra_banks.pkl'
model = pickle.load(open(MODEL_FILE_NAME, 'rb'))


@app.route('/predict')
def predict():
    """
    function for predicting house prices by client requests
    :return: predictions in JSON
    """
    crim = float(request.args.get('CRIM'))
    zn = float(request.args.get('ZN'))
    indus = float(request.args.get('INDUS'))
    chas = float(request.args.get('CHAS'))
    nox = float(request.args.get('NOX'))
    rm = float(request.args.get('RM'))
    age = float(request.args.get('AGE'))
    dis = float(request.args.get('DIS'))
    rad = float(request.args.get('RAD'))
    tax = float(request.args.get('TAX'))
    ptratio = float(request.args.get('PTRATIO'))
    b = float(request.args.get('B'))
    lstat = float(request.args.get('LSTAT'))
    record = np.array([crim, zn , indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat])
    prediction = {'predictions': model.predict(record).tolist()}
    return json.dumps(prediction)


@app.route('/predict_json', methods=['POST'])
def predict_json():
    """
    function for predicting house prices using client json input
    :return: predictions in JSON
    """
    data = pd.json_normalize(json.loads(request.get_json()))
    prediction = {'predictions': model.predict(data).tolist()}
    return json.dumps(prediction)


def main():
    port = os.environ.get('PORT')

    if port:
        # will be used if PORT is defined as environment variables (expected on Heroku)
        app.run(host='0.0.0.0', port=int(port))
    else:
        # local run
        app.run()


if __name__ == '__main__':
    main()
