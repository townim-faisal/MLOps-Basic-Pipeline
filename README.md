# MLOps Pipeline for Cat vs Dog Dataset

Currently, this repo is following Stage 1.

## Environment setup
```
virtualenv pipeline-env
pip install -r requirements.txt
```

## DVC 
```
dvc remote add -d myremote gdrive://<folder>
dvc push <file_name.dvc> -r myremote
dvc pull <file_name.dvc> -r myremote
```

## MLflow
Will run the mlflow server in a central server.
