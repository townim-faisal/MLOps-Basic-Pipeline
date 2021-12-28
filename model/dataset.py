import torch 
from torch.utils.data import Dataset
from PIL import Image 
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir 
        self.transform = transform
        
        if train:
            self.training_file = os.path.join(self.root_dir, "train")
            self.file_list = os.listdir(self.training_file)
        else: 
            self.training_file = os.path.join(self.root_dir, "val")
            self.file_list = os.listdir(self.training_file)
        
        self.transform = transform
        
        
    #dataset length
    def __len__(self):
        return len(self.file_list)
    
    #load an one of images
    def __getitem__(self,idx):
        img_path =  self.file_list[idx]
        img = Image.open(os.path.join(self.training_file, img_path))
        img_transformed = self.transform(img)
        label = img_path.rsplit('.')[0]
        
        if label == 'dog':
            label=1
        elif label == 'cat':
            label=0
            
        return img_transformed, label