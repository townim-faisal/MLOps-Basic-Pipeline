import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import yaml
import sys
import logging
from tqdm import tqdm
import json
from datetime import datetime
import mlflow

from dataset import CustomDataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from augment import transform_test

from test import Test

logging.warning("Warning. ")
tf.executing_eagerly()
# Configuration
config_file = open("config.yaml", "r")
config = yaml.safe_load(config_file)
config_file.close()

# hyperparameter
hyp_file = open("hyp.yaml", "r")
hyp = yaml.safe_load(hyp_file)
hyp_file.close()


print('Configuration:', config)
print("Hyperparameters:", hyp)

def generate_test_dataset(dir):
    #test
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    test_generator = test_datagen.flow_from_directory(dir,
                                                        target_size=(227, 227),
                                                        color_mode="rgb",
                                                        batch_size=hyp['batch_size'],
                                                        seed=hyp['seed'],
                                                        shuffle=True,
                                                        class_mode="sparse"
                                                        )
    return test_generator

# get the original_dataset
test_loader = generate_test_dataset(config['data_dir'])

print("Number of testing samples = ",len(test_loader))

test = Test(test_loader)



# MLflow on localhost with Tracking Server
mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
print("Current tracking uri:", mlflow.get_tracking_uri())
experiment_id = mlflow.set_experiment(experiment_name=config['mlflow_experiment_name'])

best_acc, best_epoch = 0, 0
SAVED_MODEL_PATH = os.path.join(config['artifact_path'], config['mlflow_experiment_name'], config['mlflow_run_name'])
#os.makedirs(SAVED_MODEL_PATH, exist_ok=True)
model_name="{}_{}_dogcat".format(hyp['model'], config['version'])
# model = tf.keras.models.load_model(os.path.join(SAVED_MODEL_PATH, model_name))
model = tf.keras.models.load_model(SAVED_MODEL_PATH)
# log file
df = pd.DataFrame(columns = ['loss', 'acc'])

with mlflow.start_run(run_name=config['mlflow_run_name'], experiment_id=experiment_id.experiment_id) as run:
    mlflow.log_params(hyp)
    loss, acc = test.run(model)
    df = df.append({'loss': loss, 'acc': acc}, ignore_index = True)
    
    if acc>best_acc:
            best_acc = acc
    
    metrics = {
            "acc": acc,
            "loss": loss
        }
    
    # save log
    df.to_csv(os.path.join(SAVED_MODEL_PATH, 'log.csv'), index=False)
    mlflow.end_run()
    





