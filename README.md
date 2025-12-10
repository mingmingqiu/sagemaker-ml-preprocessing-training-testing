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




