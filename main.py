# Put the code for your API here.
import os
import pandas as pd
import pickle
import logging
import json
from pydantic import BaseModel
from fastapi import FastAPI
from starter.starter.ml.data import process_data
from starter.starter.ml.evaluation import inference
from starter.starter.ml.model import load_xgb_model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -d s3-bucket s3://nd0821-c3-starter-code/")
    if os.system("dvc pull --force") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/libd/dvc")


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

app = FastAPI()
encoder = None
lb = None

@app.on_event("startup")
async def startup_event() -> None:
    global xgb_model, encoder, lb
    xgb_model = load_xgb_model('starter/model/model.json')
    encoder = pickle.load(open("starter/data/encoder.pickle", 'rb'))
    lb = pickle.load(open("starter/data/lb.pickle", 'rb'))


@app.get("/")
async def read_root() -> json:
    return {"message": "Welcome to the API that predicts salary from census data"}


def replace_dash(column_name: str) -> str:
    return column_name.replace('_', '-')


# Class definition of the data that will be provided as POST request
class CensusDataSchema(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = replace_dash


@app.post('/predict')
async def predict(input: CensusDataSchema):
    """
    POST request that will provide sample census data and expect a prediction
    Output:
        0 or 1
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    # Read data sent as POST
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])

    logger.info(f"Input data: {input_df}")

    processed_data, _, _, _ = process_data(
        input_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    logger.info(f"SUCCESS: process_data finished")
    preds = inference(xgb_model, processed_data)
    logger.info(f"SUCCESS: predictions were generated")
    return {"result": preds.item()}
