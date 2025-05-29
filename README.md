## How to run?

    conda create -n env python=3.8 -y
    conda activate env
    pip install -r requirements.txt && pip install -e .

    uvicorn app.main:app --reload



## use http://127.0.0.1:8000/docs to check all functionality

 i have used custom app/exception.py and app/logger.py

 I have used Feed-Forward Neural Network (FFNN) Intent Classifier and Tf-Idf

## The CI/CD pipeline has not been implemented using a .github/workflows/ci-cd.yml file with self-hosted GitHub Actions runners, as it requires AWS credentials to authenticate with services such as Amazon ECR for Docker image storage and EC2 instances for deployment.
