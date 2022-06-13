import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import yaml
import sys
import logging

from tqdm import tqdm
import tensorflow as tf

class Val:
    def __init__(self, testloader):
        self.testloader = testloader
    
    def run(self, model, loss_fn):
        # with torch.no_grad():
        #     model.eval()
            # epoch_loss = 0.0
            epoch_loss_avg = tf.keras.metrics.Mean()
            # correct = 0
            # total = 0
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            bar = tqdm(enumerate(self.testloader), total=len(self.testloader), desc="Val: ")
            for batch_id, data in bar:
                inputs, labels = data[0], data[1]
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                epoch_loss_avg.update_state(loss) 
                # epoch_loss += float(loss.item())
                # _, predicted = torch.max(outputs.data, 1)
                prediction = tf.math.argmax(outputs, axis=1, output_type=tf.int64)
                test_accuracy(prediction, labels)

                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                bar.set_postfix_str('Loss='+str(round(epoch_loss_avg.result(), 4))+', Accuracy:'+str(round(test_accuracy.result(), 4)))

            return epoch_loss_avg.result(), round(test_accuracy.result(), 4) #/len(bar), round(100*correct/total, 4)
    
    