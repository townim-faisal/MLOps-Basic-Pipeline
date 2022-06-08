import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
import yaml
import sys
import logging
from tf.keras.losses import SparseCategoricalCrossentropy 

class Trainer:
    def __init__(self, train_dataset, device):
        self.device = device
        self.train_dataset = train_dataset
    
    def loss(self, model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        loss_object = SparseCategoricalCrossentropy(from_logits=True)

        return loss_object(y_true=y, y_pred=y_)

        # l = loss(model, features, labels, training=False)
        # print("Loss test: {}".format(l))
    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, inputs, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def run(self, model, callback_list, optimizer):
        # model.train()
        # epoch_loss = 0
        # bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train: ")
        
        # for batch_id, data in bar:
        #     inputs, labels = data[0].to(self.device), data[1].to(self.device)
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     epoch_loss += float(loss.item())
        #     bar.set_postfix_str('Loss='+str(round(epoch_loss/(batch_id+1), 4)))
        #     if batch_id==10:
        #         break
        # set the callbacks
        
        # start training
        # model.fit(self.train_dataset,
        #     epochs= epochs,
        #     steps_per_epoch= self.train_dataset.samples // batch_size,
        #     validation_data=self.valid_dataset,
        #     validation_steps= self.valid_dataset.samples // batch_size,
        #     callbacks=callback_list,
        #     verbose=1)
        # Keep results for plotting
        train_loss_results = []
        train_accuracy_results = []

        for epoch in range(hyp['epochs']):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop - using batches of 32
            for x, y in self.train_dataset:
                # Optimize the model
                loss_value, grads = self.grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(y, model(x, training=True))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))
        return model