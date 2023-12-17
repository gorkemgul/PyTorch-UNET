import numpy as np 
from torch.utils.data import Dataset
import os 
import cv2

class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.image_dir, self.masks[index])
        image = np.array(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        mask = np.array(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY))
        mask[mask == 255] = 1.0 
    
        return image, mask 
    