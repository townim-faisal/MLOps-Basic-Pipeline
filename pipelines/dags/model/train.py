import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
import yaml
import sys
import logging
from tensorflow.keras.losses import SparseCategoricalCrossentropy 

class Trainer:
    def __init__(self, train_loader):
        # self.device = device
        self.train_loader = train_loader

        # l = loss(model, features, labels, training=False)
        # print("Loss test: {}".format(l))
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def loss(self, model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        loss_object = SparseCategoricalCrossentropy(from_logits=False)

        return loss_object(y_true=y, y_pred=y_)

    def run(self, model, optimizer):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train: ")
        
        for batch_id, data in bar:
            inputs, labels = data[0], data[1]
            print(batch_id)
            # start training
            loss, grads = self.grad(model, inputs, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss)  
            # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(labels, model(inputs, training=True))
            bar.set_postfix_str('Loss='+str(tf.keras.backend.get_value(epoch_loss_avg.result))) #/(batch_id+1)
            # if batch_id==10:
            #     break
        return model, optimizer, tf.keras.backend.get_value(epoch_loss_avg.result()), tf.keras.backend.get_value(epoch_accuracy.result()) #/len(bar)