import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import yaml
import sys
import logging

from tqdm import tqdm


class Val:
    def __init__(self, testloader, device):
        self.testloader = testloader
        self.device = device
    
    def run(self, model, criterion):
        with torch.no_grad():
            model.eval()
            epoch_loss = 0.0
            correct = 0
            total = 0
            bar = tqdm(enumerate(self.testloader), total=len(self.testloader), desc="Val: ")
            for batch_id, data in bar:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_loss += float(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                bar.set_postfix_str('Loss='+str(round(epoch_loss/(batch_id+1), 4))+', Accuracy:'+str(round(100*correct/total, 4)))

            return epoch_loss/len(bar), round(100*correct/total, 4)
    
    