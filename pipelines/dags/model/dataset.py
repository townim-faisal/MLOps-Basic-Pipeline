from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import math

# class CustomDataset(Dataset):
#     def __init__(self, root_dir, train=True, transform=None):
#         super(CustomDataset, self).__init__()
#         self.root_dir = root_dir 
#         self.transform = transform
        
#         if train:
#             self.training_file = os.path.join(self.root_dir, "train")
#             self.file_list = os.listdir(self.training_file)
#         else: 
#             self.training_file = os.path.join(self.root_dir, "val")
#             self.file_list = os.listdir(self.training_file)
        
#         self.transform = transform
        
        
#     #dataset length
#     def __len__(self):
#         return len(self.file_list)
    
#     #load an one of images
#     def __getitem__(self,idx):
#         img_path =  self.file_list[idx]
#         img = Image.open(os.path.join(self.training_file, img_path))
#         img_transformed = self.transform(img)
#         label = img_path.rsplit('.')[0]
        
#         if label == 'dog':
#             label=1
#         elif label == 'cat':
#             label=0
            
#         return img_transformed, label

class CustomDataset(Sequence):
    
    def __init__(self, root_dir, batch_size, train=True, transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.batch_size = batch_size
        self.label_list = []

        if train:
            self.training_file = os.path.join(self.root_dir, "train")
            self.file_list = os.listdir(self.training_file)

        else: 
            self.training_file = os.path.join(self.root_dir, "val")
            self.file_list = os.listdir(self.training_file)
        
        for img_path in self.file_list:
            if img_path.rsplit('.')[0] == 'dog':
                self.label_list.append(1)
            else:
                self.label_list.append(0)

        self.transform = transform

    def __len__(self):
        return math.ceil(len(self.file_list) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.file_list[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.label_list[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            resize(imread(file_name), (227, 227))
               for file_name in batch_x]), np.array(batch_y)