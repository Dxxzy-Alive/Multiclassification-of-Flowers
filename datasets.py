import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transforms import transform, train_transform, val_transform

data_dir =r"C:\Users\user\Documents\Computer Vision\Multiclassification\flower_photos"
batch_size = 32
g = torch.Generator().manual_seed(42)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size  = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size,val_size], generator=g)
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
val_dataloader =DataLoader(val_dataset, batch_size=batch_size, generator=g)
