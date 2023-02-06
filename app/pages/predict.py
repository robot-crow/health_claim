from flask import Blueprint, request
from headers.model_header import ModelHousing
import pandas as pd
import pickle
import json

predict_page = Blueprint('predict_page', __name__)

@predict_page.route("/predict",methods=["POST"])
def predict():
    req_j = request.json
    claim_status = pickle.load(open('models/status_housing.pkl','rb'))
    claim_num = pickle.load(open('models/claimnum_housing.pkl', 'rb'))
    claim_gross = pickle.load(open('models/gross_housing.pkl', 'rb'))

    input_data = {}
    for k, v in req_j['policy_data'].items():
        input_data.update(v)

    input_data = pd.DataFrame.from_records([input_data])

    status_pred = claim_status.make_prediction(input_data)
    claimnum_pred = claim_num.make_prediction(input_data)
    gross_pred = claim_gross.make_prediction(input_data)

    print(status_pred)
    print(claimnum_pred)
    print(gross_pred)
    return "<h1>Predict!</h1>"