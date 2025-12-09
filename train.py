import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import boto3
import os
import cloudpickle

print("here training")

input_path = "/opt/ml/input/data/train/bank-additional-processed.csv"
model_output_path = "/opt/ml/model/model.pkl"

for root, dirs, files in os.walk("/opt/ml/input/"):
    for file in files:
        print(os.path.join(root, file))

# Load processed data
df = pd.read_csv(input_path)

print(df.keys())
print(df.head())

df.rename(columns={"y_yes": "y"}, inplace=True)

# Train-test split
X = df.drop(columns=["y"])
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# # Save the model
# joblib.dump(model, "/opt/ml/model/model.pkl")

# ✅ Save model along with feature names
feature_names = X_train.columns.tolist()
# joblib.dump((model, feature_names), model_output_path)

with open(model_output_path, "wb") as f:
    cloudpickle.dump((model, feature_names), f)

print("✅ Model saved to:", model_output_path)