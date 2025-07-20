# MLOps

This project trains a machine learning model and provides predictions via an API.

## Files

- `FastAPI_pkl.py`: Runs the Backend API, which makes predictions based on the pre-trained pickle file (Isolation Forest Model)
- `MLflow_Fraud_Detection.py`: Trains the Isolation Forest Model and saves the run in a MLFLow Experiment
- `model.pkl`: pre-trained Isolation Forest Model
- `requirements.txt`: Necessary Libraries and Packages
- `REST_AI_UI`: Streamlit UI to perform Predictions

## Usage
#### CMD
```
uvicorn FastAPI_pkl.py --host 0.0.0.0 --port 8000
streamlit run REST_API_UI.py
```
