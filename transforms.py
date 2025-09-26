import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader, desc="Computing mean and std", leave=False):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum/num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean, std

temp_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

batch_size = 32
g = torch.Generator().manual_seed(42)
temp_dataset = datasets.ImageFolder(root=r"C:\Users\user\Documents\Computer Vision\Multiclassification\flower_photos", transform=temp_transform)
temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True, generator=g)
mean, std = get_mean_std(temp_loader)

mean, std = mean.tolist(), std.tolist()

class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

transform = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),         # random crop + resize
    transforms.RandomHorizontalFlip(),         # flip left-right
    transforms.RandomRotation(20),             # small random rotation
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), # change brightness/contrast/saturation
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ✅ Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])