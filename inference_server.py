from flask import Flask, request
import pickle
import pandas as pd
import json


app = Flask(__name__)
MODEL_FILE_NAME = 'tyra_banks.pkl'
model = pickle.load(open(MODEL_FILE_NAME, 'rb'))


@app.route('/predict')
def predict():
    """
    function for predicting house prices by client requests
    :return: predictions in JSON
    """
    data = pd.json_normalize(json.loads(request.args.get('json')))
    prediction = {'predictions': model.predict(data).tolist()}
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
    app.run()


if __name__ == '__main__':
    main()
