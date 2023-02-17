from flask import Blueprint, request
from headers.model_header import ModelHousing
import pandas as pd
import pickle
import json

predict_page = Blueprint('predict_page', __name__)

@predict_page.route("/predict",methods=["POST"])
def predict():
    # Write this imperative style, not iterative style
    #get a request
    req_j = request.json

    #load models
    claim_status = pickle.load(open('models/status_housing.pkl','rb'))
    claim_num = pickle.load(open('models/claimnum_housing.pkl', 'rb'))
    claim_amount = pickle.load(open('models/amount_housing.pkl', 'rb'))

    claim_num_outpat = pickle.load(open('models/outpat_claimnum_housing.pkl', 'rb'))
    claim_amount_outpat = pickle.load(open('models/outpat_amount_housing.pkl', 'rb'))

    claim_num_inpat = pickle.load(open('models/inpat_claimnum_housing.pkl', 'rb'))
    claim_amount_inpat = pickle.load(open('models/inpat_amount_housing.pkl', 'rb'))

    input_data = {}
    for k, v in req_j['policy_data'].items():
        input_data.update(v)

    input_data = pd.DataFrame.from_records([input_data])

    #will they claim?
    status_pred = claim_status.make_prediction(input_data)

    #how much, and how many times (lifetime)
    claimnum_pred = claim_num.make_prediction(input_data)
    amount_pred = claim_amount.make_prediction(input_data)

    #break down by in and outpatient
    outpat_claimnum_pred = claim_num_outpat.make_prediction(input_data)
    outpat_amount_pred = claim_amount_outpat.make_prediction(input_data)
    inpat_claimnum_pred = claim_num_inpat.make_prediction(input_data)
    inpat_amount_pred = claim_amount_inpat.make_prediction(input_data)


    print(status_pred)
    print(claimnum_pred)
    print(amount_pred)

    print(inpat_amount_pred)
    print(inpat_claimnum_pred)

    print(outpat_amount_pred)
    print(outpat_claimnum_pred)

    return "<h1>Predict!</h1>"