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
| `/opt/program/`       | Inference code (e.g., `serve.py`)  | Custom inference containers |
| `/opt/ml/code/`       | Training code (e.g., `train.py`)   | Script mode training jobs   |
| `/opt/ml/model/`      | Output model artifacts (`.tar.gz`) | After training              |
| `/opt/ml/processing/` | Data processing input/output       | Processing jobs             |
