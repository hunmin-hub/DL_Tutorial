import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import os

class CellDataset(data.Dataset):
    def __init__(self,file_path,transform=None):
        self.file_path=file_path
        self.transform=transform

        list_file=os.listdir(self.file_path)
        self.list_label=[f for f in list_file if f.startswith('label')]
        self.list_image=[f for f in list_file if f.startswith('image')]

        self.list_label.sort()
        self.list_image.sort()
    
    def __len__(self):
        return len(self.list_label)
    
    def __getitem__(self,index):
        label=np.load(os.path.join(self.file_path,self.list_label[index]))
        image=np.load(os.path.join(self.file_path,self.list_image[index]))

        label=label/255.0
        image=image/255.0

        if label.ndim==2: label=label[:,:,np.newaxis]
        if image.ndim==2: image=image[:,:,np.newaxis]

        if self.transform:
            label=self.transform(label)
            image=self.transform(image)
        
        return image,label