import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest


#Data Section
dataframe = pd.read_csv("cleaned_transactions_new_updated.csv")
le = LabelEncoder()
df = dataframe[["LongTransaction", "RepeatedLogins", "HighTransaction"]]
dataframe.drop('Unnamed: 0', axis=1, inplace=True)
dataframe["dummy_CustomerID"]=le.fit_transform(dataframe["dummy_CustomerID"])
dataframe["dummy_Location"]=le.fit_transform(dataframe["dummy_Location"])
dataframe["dummy_Channel"]=le.fit_transform(dataframe["dummy_Channel"])
dataframe["dummy_TransactionDate"] = pd.to_datetime(dataframe["dummy_TransactionDate"])
dataframe['year'] = dataframe['dummy_TransactionDate'].dt.year
dataframe['month'] = dataframe['dummy_TransactionDate'].dt.month
dataframe['day'] = dataframe['dummy_TransactionDate'].dt.day
dataframe.drop('dummy_TransactionDate', axis=1, inplace=True)
dataframe = dataframe.copy()
tempdf = dataframe.select_dtypes(include=['number'])
temp1 = df.select_dtypes(include=['number'])

#Training Section 
model_params = {'contamination':'auto',
                'random_state':42}
model = IsolationForest(**model_params)
model.fit(tempdf)
dataframe['iso_pred'] = model.predict(tempdf)
print(dataframe.head())


model1 = IsolationForest(**model_params)
model1.fit(temp1)
df['iso_pred'] = model1.predict(temp1)
print(df.head())

#Evaluation Section
dataframe['anomaly_score'] = model.decision_function(tempdf)
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(tempdf, dataframe['iso_pred'])
print(f"Silhouette Score: {silhouette}")
import matplotlib.pyplot as plt
plt.hist(dataframe['anomaly_score'], bins=50)
plt.title("Distribution of Anomaly Scores")
plt.show()


df['anomaly_score'] = model1.decision_function(temp1)
from sklearn.metrics import silhouette_score
silhouette1 = silhouette_score(temp1, df['iso_pred'])
print(f"Silhouette Score: {silhouette1}")
import matplotlib.pyplot as plt
plt.hist(df['anomaly_score'], bins=50)
plt.title("Distribution of Anomaly Scores")
plt.show()


#Utilities (MLflow)
import mlflow
from mlflow.models import infer_signature
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("MLflow Fraud Detection")

with mlflow.start_run(run_name="Fraud-Model (All Feeatures)"):
    mlflow.log_params(model_params)
    mlflow.log_metric("Silhouette Score", silhouette)
    model_signature = infer_signature(tempdf, model.predict(tempdf))
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="Isolation Forest",
        signature=model_signature,
        input_example=tempdf,
        registered_model_name="Fraud-Transactions-Clustering",
    )

    mlflow.set_tags({"Training Info": "Basic Isolation Forest Model for Clustering Fraud Data into Normal/Anomalous Transactions"}
    )