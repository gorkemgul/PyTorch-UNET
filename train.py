import torch 
import torch.nn as nn 
from torchvision.transforms import transforms
import torch.optim as optim
from tqdm import tqdm 
from model import unet
import yaml

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders, 
    check_accuracy,
    save_preds_as_images
)

class UNETTrainer():
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(**kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            
        return cls(**yaml_data)
    
    def train_step(self, loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(loader)
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(self.DEVICE)
            targets = targets.float().unsqueeze(1).to(self.DEVICE)
            
            with torch.cuda.amp.autocast():
                preds = model(data)
                loss = loss_fn(preds, targets)
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss = loss.item())

    def main(self):
        train_transform = transforms.Compose(
            transforms.Resize(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            transforms.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max = 255.0
            ),
            transforms.ToTensor()
        )
        valid_transform = transforms.Compose(
            transforms.Resize(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            transforms.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max = 255.0
            ),
            transforms.ToTensor()
        )
        


if __name__ == "__main__":
    pass 
    






