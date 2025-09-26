import matplotlib.pyplot as plt
from datasets import train_dataset, val_dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter



def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})

train_series = class_counts(train_dataset)
val_series = class_counts(val_dataset)

plt.figure(figsize=(10,5))
train_series.plot(kind="bar")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Train dataset")
plt.show();

plt.figure(figsize=(10,5))
val_series.plot(kind="bar")
plt.xlabel("Classes")
plt.ylabel("Frequency")
plt.title("Validation dataset")
plt.show();