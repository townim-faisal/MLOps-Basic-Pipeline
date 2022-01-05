import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient

with open("params/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

def print_run_info(runs):
    for r in runs:
        print("run_id: {}".format(r.info.run_id))
        print("metrics: {}".format(r.data.metrics))

def get_best_run(runs):
    best_r = None
    best_m = 0
    for r in runs:
        m = r.data.metrics
        if 'val acc' in m:
            if m['val acc']>best_m:
                best_m = m['val acc']
                best_r = r
    return best_r, best_val

mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
experiment_id = mlflow.set_experiment(experiment_name=config['mlflow_experiment_name'])

client = MlflowClient()
# runs = client.search_runs("my_experiment_id", "", order_by=["metrics.rmse DESC"], max_results=1)
runs = client.search_runs(experiment_id.experiment_id, "")
best_run, best_val = get_best_run(runs)
print(best_run.data.tags)

best_model_path = os.path.join(config['mlflow_experiment_name'], config['mlflow_run_name'], 'best_model.pth')
print(best_model_path)

with open(os.path.join(config['artifact_path'], config['mlflow_experiment_name'], 'best_model.txt'), 'w') as file:
    file.write(f'path\tval acc\n')
    file.write(f'{best_model_path}\t{best_val}')