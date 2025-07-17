from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np

class PredictionRequest(BaseModel):
      HighTransaction: int
      RepeatedLogins: int
      LongTransaction: int
      
      
app = FastAPI()
try:

    model = pickle.load(open(r"model.pkl", 'rb'))
except Exception as e:
    print(f"Error Loading the model: {e}")
    
    
@app.post('/predict')
def predict(request: PredictionRequest):
    input_data = np.array([[
        request.HighTransaction,
        request.RepeatedLogins,
        request.LongTransaction,

   ]])
    
    predictions = model.predict(input_data)
    mapped_pred = ["Normal" if prd==1 else "Suspecious" for prd in predictions]
    
    return  {"Prediction: ": mapped_pred}