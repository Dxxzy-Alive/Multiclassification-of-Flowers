import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from datasets import train_dataset, val_dataset, dataset, val_dataloader, train_dataloader
from torchinfo import summary

print(train_dataset)
classes = train_dataset.dataset.classes
class_to_idx = train_dataset.dataset.class_to_idx  # <-- dict mapping names → indices
print(classes)
print(class_to_idx)
distinct_classes = {x[1] for x in dataset.imgs}
print(distinct_classes)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# -----------------------------
# Define the model
# -----------------------------
model = torch.nn.Sequential()

# ✅ Convolutional + pooling layers
conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
model.append(conv1)
model.append(torch.nn.ReLU())
model.append(max_pool1)

conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
model.append(conv2)
model.append(torch.nn.ReLU())
model.append(max_pool2)

conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
max_pool3 = torch.nn.MaxPool2d(2)
model.append(conv3)
model.append(torch.nn.ReLU())
model.append(max_pool3)

model.append(torch.nn.Flatten())   # flatten feature maps
model.append(torch.nn.Dropout())   # regularization

# ✅ Dynamically compute `n_features`
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224)  # batch size = 1
    # Pass only through convolution + pooling layers (exclude Flatten + Dropout)
    features = model[:9](dummy)
    n_features = features.view(1, -1).shape[1]
print(f"Flattened feature size: {n_features}")

# ✅ Fully connected layers
linear1 = torch.nn.Linear(in_features=n_features, out_features=500)
model.append(linear1)
model.append(torch.nn.ReLU())
model.append(torch.nn.Dropout())

output_layer = torch.nn.Linear(500, 5)  # 5 flower classes
model.append(output_layer)

# Move model to device
model.to("cpu")

# ✅ Print model summary
height, width = 224, 224
summary(model, input_size=(32, 3, height, width))
