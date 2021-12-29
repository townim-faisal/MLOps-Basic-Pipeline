# Model Pipeline

One experiment will be conducted for one version of data in `storage\data-lake`.
One experiment can have many versions of run through changing hyperparameter in [hyp.yaml](params/hyp.yaml).
[best.py](best.py) file will help to get the best model with hyperparameter for a given experiment key `mlflow_experiment_name` in [config.yaml](params/config.yaml).

## MLFlow
Run mlflow in root folder.
```bash
mlflow ui
```