# Use an official SageMaker PyTorch image as the base
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39-ubuntu20.04-sagemaker

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Install additional Python dependencies
RUN pip install --upgrade pip && pip install pandas numpy scikit-learn boto3 s3fs

# Copy preprocessing and training scripts into the container
COPY preprocess.py /opt/ml/code/preprocess.py
COPY train.py /opt/ml/code/train.py
COPY test.py /opt/ml/code/test.py

# Set entrypoint script (to allow for flexibility)
ENTRYPOINT ["python", "train.py"]
