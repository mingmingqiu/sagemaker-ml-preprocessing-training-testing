import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import boto3

bucket_name = "feature-engineering-bucket-989220949c9c"
input_path = "/opt/ml/processing/output/bank-additional-processed.csv"
model_output_key = "Models/random-forest-model.joblib"

# Load processed data
s3 = boto3.client("s3")
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
s3.upload_file("/opt/ml/model/model.joblib", bucket_name, model_output_key)
print(f"Trained model saved to s3://{bucket_name}/{ model_output_key}")
