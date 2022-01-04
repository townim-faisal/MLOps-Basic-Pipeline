# MLOps Pipeline for Cat vs Dog Dataset

Currently, this repo is following Stage 1. Read [here](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_1_ml_pipeline_automation).

```bash
app/                 - application deployment
└── app.py           - run application

data/                - data pipeline
├── params           - Parameter
    └── param.yaml   - parameter file
├── log              - log folder
├── figures          - figures folder
├── notebooks        - jupyter notebooks
├── eda.py           - exploratory data analysis
├── preprocess.py    - preprocessing data
└── validate.py      - validating data

storage/
├── artifact         - model 
├── data-lake        - raw data
└── data-warehouse   - preprocessed data 
```


## Environment setup
Python version 3.8.5 is installed.
```
python -m venv pipeline-env
pip install -r requirements.txt
```
N.B: If you are in wsl, make sure run `pip install wheel` if needed.


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
Will run the mlflow server in a central server. If you want to run `mlflow` locally, just run `mlflow ui` from root folder.

## Airflow
Use wsl or linux as there is no support for windows. Read setup in [here](pipelines/README.md).
