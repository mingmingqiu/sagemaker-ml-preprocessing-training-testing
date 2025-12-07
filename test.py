import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import boto3

bucket_name = "feature-engineering-bucket-989220949c9c"
input_key = "/opt/ml/processing/output/bank-additional-processed.csv"
model_key = "Models/random-forest-model.joblib"

# Load processed data and model
s3 = boto3.client("s3")
s3.download_file(bucket_name, model_key, "/opt/ml/model/model.joblib")
df = pd.read_csv(input_key)
model = joblib.load("/opt/ml/model/model.joblib")

# Evaluate the model
X = df.drop(columns=["y"])
y = df["y"]
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy}")
