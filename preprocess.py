import pandas as pd
import boto3

print("here preprocessing")

input_path = "/opt/ml/processing/input/bank-additional-full.csv"
output_path = "/opt/ml/processing/output/bank-additional-processed.csv"

# Load dataset
df = pd.read_csv(input_path)

# Preprocessing steps
df["age_squared"] = df["age"] ** 2
df = pd.get_dummies(df, drop_first=True)

# Rename label column
df.rename(columns={"y_yes": "y"}, inplace=True)

# Save processed data
df.to_csv(output_path, index=False)
