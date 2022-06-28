import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import yaml
import sys
import logging
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy 
from tqdm import tqdm

class Test:
    def __init__(self, testloader):
        self.testloader = testloader

    def loss(self, model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)
        loss_object = SparseCategoricalCrossentropy(from_logits=True)

        return y_, loss_object(y_true=y, y_pred=y_)

    
    def run(self, model):
        # with torch.no_grad():
        #     model.eval()
            # epoch_loss = 0.0
            epoch_loss_avg = tf.keras.metrics.Mean()
            # correct = 0
            # total = 0
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            bar = tqdm(enumerate(self.testloader), total=len(self.testloader), desc="Test: ")
            for batch_id, data in bar:
                
                inputs, labels = data[0], data[1]
                
                # outputs = model(inputs)
                outputs, loss = self.loss(model, inputs, labels, training=False)
                epoch_loss_avg.update_state(loss) 
                # epoch_loss += float(loss.item())
                # _, predicted = torch.max(outputs.data, 1)
                # prediction = tf.math.argmax(outputs, axis=1, output_type=tf.int64)
                test_accuracy.update_state(labels, outputs)

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                bar.set_postfix_str('Loss='+str(tf.keras.backend.get_value(epoch_loss_avg.result()))+', Accuracy:'+str(tf.keras.backend.get_value(test_accuracy.result())))

            return tf.keras.backend.get_value(epoch_loss_avg.result()), tf.keras.backend.get_value(test_accuracy.result()) #/len(bar), round(100*correct/total, 4)
    
    