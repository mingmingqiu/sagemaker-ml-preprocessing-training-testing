import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import boto3
import os

print("here training")

input_path = "/opt/ml/input/data/training/bank-additional-processed.csv"

contents = os.listdir(input_path)
print(contents)

# Load processed data
df = pd.read_csv(input_path)

# Train-test split
X = df.drop(columns=["y"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "/opt/ml/model/model.joblib")