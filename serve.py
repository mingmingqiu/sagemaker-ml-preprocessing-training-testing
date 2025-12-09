import os
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import cloudpickle

# Load model
def model_fn(model_dir):
    print("📦 Loading model from:", model_dir)
    # model, feature_names = joblib.load(os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model, feature_names = cloudpickle.load(f)
    return model, feature_names

# Parse input request
def input_fn(request_body, content_type="text/csv"):
    print("📥 Parsing input...")
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body), header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Preprocess + predict
def predict_fn(input_data, model_and_features):
    print("🤖 Running prediction...")
    model, feature_names = model_and_features

    # Apply preprocessing (replicates what was done in preprocess.py)
    input_data.columns = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays",
        "previous", "poutcome", "emp.var.rate", "cons.price.idx",
        "cons.conf.idx", "euribor3m", "nr.employed"
    ]

    input_data["age_squared"] = input_data["age"] ** 2

    # One-hot encoding
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure same column order as training
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    preds = model.predict(input_data)
    return preds.tolist()

# Convert prediction to response
def output_fn(prediction, accept="application/json"):
    print("📤 Formatting output...")
    if accept == "application/json":
        return str(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")



# # serve.py
# import os
# import joblib
# import json
# import pandas as pd
# from io import StringIO

# import logging
# from datetime import datetime
# import uuid

# # 🔧 Setup CloudWatch logger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# # # 🔧 Optional: S3 logging config
# # ENABLE_S3_LOGGING = True
# # S3_BUCKET = "feature-engineering-bucket"
# # S3_PREFIX = "inference-logs/"
# # s3_client = boto3.client("s3")


# def model_fn(model_dir):
#     return joblib.load(os.path.join(model_dir, "model.joblib"))


# def input_fn(request_body, request_content_type):
#     if request_content_type == "text/csv":
#         return pd.read_csv(StringIO(request_body), header=None)
#     raise ValueError("Unsupported content type")


# def predict_fn(input_data, model):
#     return model.predict(input_data)


# def output_fn(prediction, response_content_type):
#     response = {
#         "timestamp": datetime.utcnow().isoformat(),
#         "predictions": prediction.tolist()
#     }

#     # ✅ Log to CloudWatch
#     # This string is directly returned as the inference result from the model endpoint. No file writing is involved.
#     logger.info(f"Inference output: {json.dumps(response)}")

#     # # ✅ Optional: save to S3
#     # if ENABLE_S3_LOGGING:
#     #     request_id = str(uuid.uuid4())
#     #     s3_key = f"{S3_PREFIX}{request_id}.json"
#     #     try:
#     #         s3_client.put_object(
#     #             Bucket=S3_BUCKET,
#     #             Key=s3_key,
#     #             Body=json.dumps(response)
#     #         )
#     #     except Exception as e:
#     #         logger.warning(f"Failed to upload inference log to S3: {e}")

#     return json.dumps(response)
    
# # def output_fn(prediction, response_content_type):
# #     # This string is directly returned as the inference result from the model endpoint. No file writing is involved.
# #     return json.dumps({"predictions": prediction.tolist()})

# # # handle multiple formats (recommended for general-purpose APIs):
# # def output_fn(prediction, response_content_type):
# #     if response_content_type == "application/json":
# #         return json.dumps({"predictions": prediction.tolist()})
# #     elif response_content_type == "text/csv":
# #         return ",".join(map(str, prediction.tolist()))
# #     else:
# #         raise ValueError(f"Unsupported response content type: {response_content_type}")
