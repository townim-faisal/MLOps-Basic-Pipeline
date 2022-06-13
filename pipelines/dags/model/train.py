import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
import yaml
import sys
import logging

class Trainer:
    def __init__(self, train_loader):
        # self.device = device
        self.train_loader = train_loader

        # l = loss(model, features, labels, training=False)
        # print("Loss test: {}".format(l))
    def grad(self, model, inputs, targets, loss):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def run(self, model, loss, optimizer):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train: ")
        
        for batch_id, data in bar:
            inputs, labels = data[0], data[1]
            # start training
            loss, grads = self.grad(model, inputs, labels, loss)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss)  
            # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(labels, model(inputs, training=True))
            bar.set_postfix_str('Loss='+str(round(epoch_loss_avg/(batch_id+1), 4)))
            if batch_id==10:
                break
        return model, optimizer, epoch_loss_avg.result() #/len(bar)