<h1>Introduction</h1>
The whole popeline is in `ml-preprocessing-training-test.ipynb` containing three jobs: preprocessing, training and testing.

<h2>1. Processing Jobs support arbitrary scripts + arbitrary input/output paths</h2>
Processing jobs (via ScriptProcessor, Processor, ProcessingStep) are designed for:

- ETL
- Data prep
- Feature engineering
- Merging datasets
- Cleaning
- Validation

These jobs allow:

✔ any directory layout
✔ arbitrary scripts
✔ multiple named inputs and outputs
✔ mapping each S3 input → container destination

Example:

```bash
ProcessingInput(
    source="s3://bucket/data.csv",
    destination="/opt/ml/processing/input"
)
```

This works because processing jobs mount everything you specify.


<h2>2. Training Jobs must follow the built-in SageMaker training contract</h2>
Training jobs (via Estimator, TrainingStep) follow a strict pattern that AWS requires for compatibility with:

- Metrics logging
- distributed training
- checkpoint saving
- model artifacts export
- hyperparameter tuning
- managed training containers
- Script Mode execution
  
Training jobs do NOT support arbitrary ProcessingInput/ProcessingOutput.

They instead expect:

✔ A single training dataset
✔ Optional validation dataset
✔ Output model placed automatically in:

```bash
/opt/ml/model
```

<h2>3. Training jobs use a fixed input interface: TrainingInput(...)</h2>
You provide only the S3 URI, not the container path:

```bash
inputs={
    "train": TrainingInput(
        s3_data="s3://bucket/path/train_data.csv"
    )
}
```
Inside the container, SageMaker **always maps it automatically** to:

```bash
/opt/ml/input/data/train/train_data.csv
```
You cannot change this path.

This is part of the SageMaker Training contract.

<h2>4. Training jobs also require a fixed output interface</h2>
Models MUST be written to:

```bash
/opt/ml/model
```

Anything you save here will be automatically uploaded to:
```bash
s3://<bucket-name>/models/<pipeline-name>/output/model.tar.gz
```

<h2>5. Inference</h2>
Serve.py for inference should be copied in Dockerfile

because SageMaker Inference Containers expect the inference entry point to be located in /opt/program/serve.py, not /opt/ml/code/.

<h2>6. Paths and their purposes</h2>

| Path                  | Used For                           | When?                       |
| --------------------- | ---------------------------------- | --------------------------- |
| `/opt/ml/code/`       | Inference code (e.g., `serve.py`)  | Custom inference containers |
| `/opt/ml/code/`       | Training code (e.g., `train.py`)   | Script mode training jobs   |
| `/opt/ml/model/`      | Output model artifacts (`.tar.gz`) | After training              |
| `/opt/ml/processing/` | Data processing input/output       | Processing jobs             |

<h2>7. Some jobs definition</h2>

- SageMaker Endpoint for real-time inference
- Model Registry for version control and approval

<h2>8. Solutions to some questions</h2>
1). Why Register Model?

- You don’t need to register a model to deploy it.
- But model registration is useful if:
- You want model versioning (each trained model has a version in a group).
- You want model approval workflow (e.g., "Approved", "Pending").
- You use Model Registry in CI/CD pipelines.

2). Why Deployment Works in One Script but Not the Pipeline

Because:

In the pipeline: model artifact might not yet be fully written to S3 when you immediately try to deploy.

In the separate script: the model is already available in S3.

You should always ensure model path is fully available. You're doing the right thing with this:

`execution.wait()  # ✅ Waits for the pipeline to complete`

But also ensure that the `step["Metadata"]["TrainingJob"]["ModelArtifacts"]["S3ModelArtifacts"]` exists

3). We must ensure the sklearn versions are the same in both Dockerfile-training and Dockerfile-Inference

When building the docker image `mlops-train` and `mlops-inference`, we can do the following to get the version of scikit-learn:

```bash
docker run -it mlops-train /bin/bash
```
This allows to open an interactive container, then, we can do the following command within the container:
```bash
python -c "import sklearn; print(sklearn.__version__)"
```

3). ClientError: An error occurred (ValidationException) when calling the CreateDataQualityJobDefinition operation: 
Endpoint 'mlops-prod-endpoint-29' does not exist or is not valid

Model Monitor cannot attach itself to an endpoint unless the endpoint enables capture.

4). Why can’t we run inference directly from a Model in the Model Registry?
Because a Model Package / Model Registry entry contains only metadata + model files \
➡️ It does NOT contain compute resources to execute inference. \
A Model Package is essentially:
```bash
- Model.tar.gz (your model)
- Container image URI
- Input/output formats
- Version metadata
```
But no:

- CPU/GPU
- Memory
- Networking
- Web server
- /invocations API

Therefore, the registry cannot execute predict() on its own. \
It is a blueprint, not a running service. 

5). ✅ Why do we create an Endpoint?
A SageMaker Endpoint:
- Launches EC2 compute (e.g., ml.m5.large)
- Loads your container (inference Docker image)
- Loads your model artifacts (model.tar.gz)
- Runs them 24/7
- Exposes /invocations as a REST API
- Handles autoscaling, retries, logging

Without an endpoint, there is no runtime environment to execute a prediction.

6). Difference between event bridge nad lambda function

Use EventBridge when you need to define “WHEN”: \
⏰ Scheduled retraining 
- Every night
- Every week
- Every month
  
→ EventBridge Schedule Rule triggers Lambda.

📦 Event-driven retraining: 
- S3 receives new training data
- Model Monitor detects drift
- Model Registry receives an approved model
→ EventBridge detects the event → triggers Lambda.

Use Lambda when you need logic or action (“WHAT to do”) \
Examples of logic inside Lambda:

▶ Start a SageMaker Pipeline
```bash
sm.start_pipeline_execution(PipelineName="MLOpsPipeline")
```
▶ Update endpoint to latest model
```bash
sm.update_endpoint(...)
```
▶ Approve a model automatically
```bash
sm.update_model_package(ModelApprovalStatus="Approved")
```

▶ Check data drift metric before retraining \
Lambda can inspect S3 monitor reports → decide whether to retrain.

🧩 How They Work Together in MLOps 
```java
EventBridge (WHEN) 
     ↓ triggers 
Lambda (WHAT) 
     ↓ executes 
SageMaker Pipeline / Endpoint / Registry 
```

Example: Nightly retraining
```bash
EventBridge cron(0 2 * * ? *)  →  Lambda  →  sm.start_pipeline_execution()
```

<h1> Install new kernels</h1>
if the sagemaker version is not correct and we have the difficulty of installing the latest sagemaker version, then, we can create a new kernel function:

Create an isolated venv in Jupyter and install v2 inside it. Run the following code:
```bash
python -m venv sm-v2
source sm-v2/bin/activate
pip install "sagemaker==2.215.0"
```
Then start Jupyter using that venv kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name sm-v2 --display-name "Python (SageMaker v2)"
```
Now switch your notebook kernel to Python (SageMaker v2).




<h1> Full worked MLOps Pipeline</h1>

define the following codes named `lambda_function.py` in `mlops-start-pipeline` lambda step function

```bash
import boto3
import os

sm = boto3.client("sagemaker")

# Name of your existing SageMaker Pipeline
PIPELINE_NAME = "MLOpsFullPipeline"

def lambda_handler(event, context):
    """
    This Lambda is triggered every hour by EventBridge.
    It starts the SageMaker Pipeline execution automatically.
    """
    
    print("🔥 LAMBDA HANDLER PIPELINE STARTED")

    print(f"Trigger event: {event}")

    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName="HourlyRetrain",
    )

    print("Started pipeline:", response["PipelineExecutionArn"])

    return {
        "status": "Pipeline triggered",
        "execution_arn": response["PipelineExecutionArn"]
    }

```

define the following codes named `lambda_function.py` in `mlops-endpoint-deployer` lambda step function

```bash
import boto3
import botocore
import datetime

sm = boto3.client("sagemaker")

def lambda_handler(event, context):
    print("🔥 LAMBDA HANDLER STARTED")
    print("Received event:", event)

    model_package_arn = event["model_package_arn"]
    endpoint_name = event["endpoint_name"]

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # Create a new unique model for every deployment
    model_name = f"{endpoint_name}-model-{timestamp}"
    config_name = f"{endpoint_name}-config-{timestamp}"

    # -------------------------------------------------------
    # 1. Create Model (always new)
    # -------------------------------------------------------
    print(f"Creating model: {model_name}")

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "ModelPackageName": model_package_arn
        },
        ExecutionRoleArn="arn:aws:iam::961807745392:role/datazone_usr_role_bx42b1pkqcgp35_bth08ij7nkdhq9"
    )

    # -------------------------------------------------------
    # 2. Create new endpoint config (always new)
    # -------------------------------------------------------
    print(f"Creating endpoint config: {config_name}")

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
                "ModelName": model_name,   # <--- NEW model
                "VariantName": "AllTraffic"
            }
        ]
    )

    # -------------------------------------------------------
    # 3. Create / Update Endpoint
    # -------------------------------------------------------
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except botocore.exceptions.ClientError:
        endpoint_exists = False

    if not endpoint_exists:
        print(f"Creating endpoint: {endpoint_name}")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    else:
        print(f"Updating endpoint: {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    print("Deployment triggered successfully.")

    return {
        "status": "OK",
        "endpoint_name": endpoint_name,
        "endpoint_config": config_name,
        "model_name": model_name,
        "model_package_arn": model_package_arn
    }

```

The whole worked pipeline codes are the codes below `Worked version including the whole MLOps pipeline` of `ml-preprocessing-training-test.ipynb`

the retraining codes are in `Regular retraining every 2am in the morning` of `ml-preprocessing-training-test.ipynb`

the real time inference are in `How This Pipeline Supports REAL-TIME INFERENCE` of `ml-preprocessing-training-test.ipynb`
