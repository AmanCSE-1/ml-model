from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import joblib


# Creating FastAPI instance
app = FastAPI()


class Item(BaseModel):
    # ad_feature
    banner_pos: int
    airtime: int
    airlocation: int

    # user features
    relationship_status: int
    targeted_sex: int
    site_domain: int
    device_type: int
    hour_of_day: int

    # company feature
    industry: int
    genre: int

    # product feature
    expensive: int
    money_back_guarantee: int


with open("XGB Classifier CTR Prediction", "rb") as f:
    model = joblib.load(f)


@app.get('/')
def root():
    return {'status': 'SUCCESS',
            'message': 'This is a root directory'}


# Creating an Endpoint to receive the data to make prediction on.
@app.post("/predict/")
def predict(input_data: Item):
    received = input_data.dict()
    banner_pos = received['banner_pos']
    airtime = received['airtime']
    airlocation = received['airlocation']
    relationship_status = received['relationship_status']
    targeted_sex = received['targeted_sex']
    site_domain = received['site_domain']
    device_type = received['device_type']
    hour_of_day = received['hour_of_day']
    industry = received['industry']
    genre = received['genre']
    expensive = received['expensive']
    money_back_guarantee = received['money_back_guarantee']

    prediction_result = model.predict([[banner_pos, site_domain, device_type, relationship_status, industry, genre,
                                        targeted_sex, airtime, airlocation, expensive, money_back_guarantee,
                                        hour_of_day]])

    # Return the Result
    return {'status': 'SUCCESS',
            'message': 'Prediction is made successfully',
            'CTR_result': int(prediction_result)}
