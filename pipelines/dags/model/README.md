# Model Pipeline

One experiment will be conducted for one version of data in `storage\data-lake`.
One experiment can have many versions of run through changing hyperparameter in [hyp.yaml](params/hyp.yaml).
[best.py](best.py) file will help to get the best model with hyperparameter for a given experiment key `mlflow_experiment_name` in [config.yaml](params/config.yaml).

## MLFlow
If you want to track locally, run mlflow in main project folder where mlruns folder is situated.
```bash
mlflow ui
```

## Configuration File
A sample configuration file is in [here](params/config.yaml).
```
data_dir: <absolute path of data warehouse's dataset directory>
num_classes: <number of classes need to be predicted>
class_names: <array of class names>
# mlflow
mlflow_experiment_name: <experiment name>
mlflow_run_name: <naem of version of the experiment>
mlflow_tracking_uri: <path or url of mlflow> # 'file:/mnt/c/T.Faisal/OFFICE/mlops-pipeline/mlruns'
artifact_path: <absolute path where artifact and log will be saved>
```