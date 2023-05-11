from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import pandas as pd
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


with open("XGB Classifier CTR Prediction.pkl", "rb") as f:
    model = joblib.load(f)


@app.get('/')
def root():
    return {'status': 'SUCCESS',
            'message': 'This is a root directory'}

# Chart visualization function that returns pie chart data for features 
def chart_visualization(industry):
    industry_mapping = {0: 'Auto', 1: 'ClassAction', 2: 'Entertainment', 3: 'Other', 4: 'Pharma', 5: 'Political'}
    industry = industry_mapping[industry]

    site_domain_entire_data = {'Auto': {'Facebook': 20106,'Instagram': 14439,'Twitter': 9354,'LinkedIn': 6498,'Snapchat': 2923},
                               'ClassAction': {'Facebook': 1207,'Instagram': 880,'Twitter': 558,'LinkedIn': 416,'Snapchat': 197},
                               'Entertainment': {'Facebook': 5693,'Instagram': 3984,'Twitter': 2559,'LinkedIn': 1772,'Snapchat': 848},
                               'Other': {'Facebook': 18106,'Instagram': 13104,'Twitter': 8289,'LinkedIn': 5786,'Snapchat': 2635},
                               'Pharma': {'Facebook': 135785,'Instagram': 98051,'Twitter': 62546,'LinkedIn': 43351,'Snapchat': 19900},
                               'Political': {'Facebook': 2730,'Instagram': 2070,'Twitter': 1277,'LinkedIn': 913,'Snapchat': 414}}

    site_domain_graph_data = site_domain_entire_data[industry]
    site_domain_graph_data = [{'name': k, 'value': v} for k, v in site_domain_graph_data.items()]


    genre_entire_data = {'Auto': {'Comedy': 47299,'Infomercial': 3840,'Drama': 1540,'Other': 434,'Direct': 207},
                         'ClassAction': {'Comedy': 2210, 'Drama': 420, 'Infomercial': 412, 'Other': 216},
                         'Entertainment': {'Comedy': 11515, 'Infomercial': 2396, 'Drama': 731, 'Other': 214},
                         'Other': {'Comedy': 41084,'Infomercial': 3389,'Drama': 2301,'Other': 843,'Direct': 303},
                         'Pharma': {'Comedy': 332915,'Infomercial': 13316,'Drama': 10271,'Direct': 2280,'Other': 851},
                         'Political': {'Comedy': 5929, 'Drama': 753, 'Infomercial': 618, 'Other': 104}}

    genre_graph_data = genre_entire_data[industry]
    genre_graph_data = [{'name': k, 'value': v} for k, v in genre_graph_data.items()]

    # user feature
    targeted_sex_entire_data = {'Auto': {'Male': 31387, 'Female': 21933},
                         'ClassAction': {'Male': 2102, 'Female': 1156},
                         'Entertainment': {'Female': 9305, 'Male': 5551},
                         'Other': {'Female': 47920, 'Male':18076},
                         'Pharma': {'Male': 359633, 'Female': 67435},
                         'Political': {'Male': 4397, 'Female': 3007}}

    targeted_sex_graph_data = targeted_sex_entire_data[industry]
    targeted_sex_graph_data = [{'name': k, 'value': v} for k, v in targeted_sex_graph_data.items()]

    graph_data_array = [{'site_domain': site_domain_graph_data,
                         'genre': genre_graph_data,
                         'targeted_sex': targeted_sex_graph_data}]

    return graph_data_array


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

    prediction_result = model.predict_proba([[banner_pos, site_domain, device_type, relationship_status, industry, genre,
                                        targeted_sex, airtime, airlocation, expensive, money_back_guarantee,
                                        hour_of_day]])
    
    # Multiply by 100, Rounded off ctr upto 2 dec places, set range between 2-72
    ctr_result_adjustment = str(round(max(0.5, min(72.14, float(str(prediction_result[0][1]))*100)), 2))
    
    # Call the graph visualization function
    graph_data_array = chart_visualization(industry)
    
    # Return the Result
    return {'status': 'SUCCESS',
            'message': 'Prediction is made successfully',
            'CTR_result': ctr_result_adjustment,
            'graph_visualization': graph_data_array}
