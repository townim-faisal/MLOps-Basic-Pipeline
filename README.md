# MLOps Pipeline for Cat vs Dog Dataset

Currently, this repo is following Stage 1. Read [here](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_1_ml_pipeline_automation).

## Environment setup
```
virtualenv pipeline-env
pip install -r requirements.txt
```


## GCloud
Install GCloud SDK from [here](https://cloud.google.com/sdk/docs/install).
```
gcloud auth login
gcloud auth application-default login
```

## DVC 
```
dvc remote add -d myremote gs://ml-pipeline-demo-storage
dvc push <file_name.dvc> -r myremote
dvc pull <file_name.dvc> -r myremote
```

## MLflow
Will run the mlflow server in a central server.
