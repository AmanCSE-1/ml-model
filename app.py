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
    url = 'https://drive.google.com/file/d/1I4ZFa3weBOozdAitMOHVvopOVvbyzvHm/view?usp=sharing'
    url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

    df = pd.read_csv(url)

    industry_mapping = {0: 'Comedy', 1: 'Direct', 2: 'Drama', 3: 'Infomercial', 4: 'Other'}
    industry = industry_mapping[industry]
    df = df[(df['industry'] == industry) & (df['click'] == 1)]

    # advertisement feature
    banner_pos_graph_data = df['banner_pos'].value_counts().to_dict()
    banner_pos_graph_data = [{'name': k, 'value': v} for k, v in banner_pos_graph_data.items()]

    site_domain_graph_data = df['site_domain'].value_counts().to_dict()
    site_domain_graph_data = [{'name': k, 'value': v} for k, v in site_domain_graph_data.items()]

    genre_graph_data = df['genre'].value_counts().to_dict()
    genre_graph_data = [{'name': k, 'value': v} for k, v in genre_graph_data.items()]

    # user feature
    targeted_sex_graph_data = df['targeted_sex'].value_counts().to_dict()
    targeted_sex_graph_data = [{'name': k, 'value': v} for k, v in targeted_sex_graph_data.items()]

    relationship_status_graph_data = df['realtionship_status'].value_counts().to_dict()
    relationship_status_graph_data = [{'name': k, 'value': v} for k, v in relationship_status_graph_data.items()]

    graph_data_array = [{'banner_pos': banner_pos_graph_data,
                         'site_domain': site_domain_graph_data,
                         'genre': genre_graph_data,
                         'targeted_sex': targeted_sex_graph_data,
                         'relationship_status': relationship_status_graph_data}]

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
    
    # Call the graph visualization function
    graph_data_array = chart_visualization(industry)
    
    # Return the Result
    return {'status': 'SUCCESS',
            'message': 'Prediction is made successfully',
            'CTR_result': str(prediction_result[0][1]),
            'graph_visualization': graph_data_array}
