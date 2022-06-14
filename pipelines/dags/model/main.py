import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
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
from tensorflow.keras.callbacks import ReduceLROnPlateau

# import config
import models
from utils import generate_train_dataset, loss
from augment import transform_train, transform_val
from models import AlexNet
from train import Trainer
from val import Val

logging.warning("Warning. ")
tf.executing_eagerly()
# Configuration
config_file = open("params/config.yaml", "r")
config = yaml.safe_load(config_file)
config_file.close()

# hyperparameter
hyp_file = open("params/hyp.yaml", "r")
hyp = yaml.safe_load(hyp_file)
hyp_file.close()


print('Configuration:', config)
print("Hyperparameters:", hyp)

# train_data_config = (config.train_dir, config.valid_dir, config.image_height, config.image_width, config.batch_size, config.batch_size)
print(config['data_dir'])
train_loader = CustomDataset(root_dir = config['data_dir'], batch_size = hyp['batch_size'], train = True, transform = transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size = hyp['batch_size'], shuffle = True, num_workers = hyp['workers'])

val_loader = CustomDataset(root_dir = config['data_dir'], batch_size = hyp['batch_size'], train = False, transform = transform_val)
# val_loader = torch.utils.data.DataLoader(valset, batch_size = hyp['batch_size'], shuffle = False, num_workers = hyp['workers'])
# get the original_dataset
# train_dataset, valid_dataset = generate_train_dataset(train_data_config)
# result_save_path = os.path.join(config.result_dir, config.model)

print("Number of training samples = ",len(train_loader))
print("Number of testing samples = ",len(val_loader))


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(input_shape=(config['image_height'], config['image_width'], config['num_channels']), num_classes=config['num_classes'])
# optimizer = optim.Adam(model.parameters(), lr=float(hyp['lr']), weight_decay=float(hyp['wd']))
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=hyp['lr'], decay_steps=20, decay_rate=hyp['decay_rate'])

if hyp['optimizer_fn'] == 'sgd':
    optimizer = tf.keras.optimizers.SGD(hyp['lr'], hyp['momentum'])
else:
    print("add another optimizer like Adam or RMSprop")
     #loss='categorical_crossentropy',
model.compile(optimizer= optimizer,
                loss=loss,
                metrics=['accuracy'])
trainer = Trainer(train_loader)
val = Val(val_loader)
training_log = {}


# MLflow on localhost with Tracking Server
# mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root file:/home/your_user/mlruns
mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
print("Current tracking uri:", mlflow.get_tracking_uri())
experiment_id = mlflow.set_experiment(experiment_name=config['mlflow_experiment_name'])

best_acc, best_epoch = 0, 0
SAVED_MODEL_PATH = os.path.join(config['artifact_path'], config['mlflow_experiment_name'], config['mlflow_run_name'])
os.makedirs(SAVED_MODEL_PATH, exist_ok=True)

# log file
df = pd.DataFrame(columns = ['epoch', 'lr', 'train_loss', 'val_loss', 'val_acc'])

with mlflow.start_run(run_name=config['mlflow_run_name'], experiment_id=experiment_id.experiment_id) as run:
    mlflow.log_params(hyp)
    for epoch in range(hyp['epochs']):
        print(f"Epoch: {epoch+1}/{hyp['epochs']}")
        model, optimizer, training_loss = trainer.run(model, optimizer)    
        val_loss, val_acc = val.run(model)
        lr = optimizer.lr
        df = df.append({'epoch': epoch+1, 'lr': lr, 'train_loss': training_loss, 'val_loss': val_loss, 'val_acc': val_acc}, ignore_index = True)
        """
        train loss, val loss, val acc, per class accuracy, learning rate -> store in ./log/log.csv
        """
        # save best and last model
        if val_acc>best_acc:
            best_acc = val_acc
            best_epoch = epoch+1
            model_name="{}_{}_dogcat".format(hyp['model'], config['version'])
            model.save(os.path.join(SAVED_MODEL_PATH), model_name)
            # best_artifact = {
            #     "model": model.state_dict(),
            #     "epoch": best_epoch,
            #     "accuracy": best_acc,
            #     "optimizer": optimizer.state_dict()
            # }
            # torch.save(best_artifact, os.path.join(SAVED_MODEL_PATH, 'best_model.pth'))
            print('Saved best model in:', os.path.join(SAVED_MODEL_PATH, model_name))

        # artifact = {
        #     "model": model.state_dict(),
        #     "epoch": epoch+1,
        #     "accuracy": val_acc
        # }
        # torch.save(artifact, os.path.join(SAVED_MODEL_PATH, 'last_model.pth'))
        model_name="{}_{}_dogcat_last_model".format(hyp['model'], config['version'])
        model.save(os.path.join(SAVED_MODEL_PATH, model_name))
        # mlflow log metrics
        metrics = {
            "val acc": val_acc,
            "val loss": val_loss
        }
        mlflow.log_metrics(metrics, step=epoch+1)
    
    # save log
    df.to_csv(os.path.join(SAVED_MODEL_PATH, 'log.csv'), index=False)
    mlflow.end_run()
    





