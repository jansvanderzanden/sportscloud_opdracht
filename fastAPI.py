from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import pickle
import json
from io import BytesIO


app = FastAPI()

pkl_filename = "../model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)
    

class Item(BaseModel):
    name: str

@app.get("/")
def home():
    return {"Data": "Test"}
    
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    return df.to_dict(orient='records')

#----- test hierbeneden, boven werkt! op pickle file na

@app.post('/sentiment')
async def basic_predict(request: Request):

    # Getting the JSON from the body of the request
    input_data = await request.json()

    # Converting JSON to Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Getting the prediction from the Logistic Regression model
    pred = model.(input_df)

    return pred