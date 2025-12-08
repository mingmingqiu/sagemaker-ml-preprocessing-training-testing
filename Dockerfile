# Use an official SageMaker PyTorch image as the base
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39-ubuntu20.04-sagemaker

# Set the working directory inside the container
WORKDIR /opt/ml/code

# Install additional Python dependencies
RUN pip install --upgrade pip && pip install pandas numpy scikit-learn boto3 s3fs

# ✅ Copy the serve.py to where SageMaker expects it
COPY serve.py /opt/program/serve.py

# (Optional) SageMaker expects this location for inference script
ENV SAGEMAKER_PROGRAM=serve.py
ENV PYTHONUNBUFFERED=TRUE