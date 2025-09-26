import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image