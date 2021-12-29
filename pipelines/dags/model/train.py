import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import yaml
import sys
import logging


class Trainer:
    def __init__(self, train_loader, device):
        self.device = device
        self.train_loader = train_loader
    
    def run(self, model, criterion, optimizer):
        model.train()
        epoch_loss = 0
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train: ")
        
        for batch_id, data in bar:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            bar.set_postfix_str('Loss='+str(round(epoch_loss/(batch_id+1), 4)))
            if batch_id==10:
                break
        
        return model, optimizer, epoch_loss/len(bar)