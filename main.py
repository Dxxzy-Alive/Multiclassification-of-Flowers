import os
import torch
from PIL import Image
import torch.nn.functional as F
from model import model            # <-- same model architecture
from transforms import transform   # <-- same preprocessing

# ✅ Load trained weights
model.load_state_dict(torch.load("flower_model.pth", map_location="cpu"))
model.eval()                        # set to inference mode

# Labels
labels = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# Load and preprocess an image
data_dir = r"C:\Users\user\Documents\Computer Vision\Multiclassification\flower_photos\dandelion"
img_path = os.path.join(data_dir, "10443973_aeb97513fc_m.jpg")
img = Image.open(img_path)

img = transform(img).unsqueeze(0).to("cpu")  # add batch dimension

# Make prediction
with torch.no_grad():
    outputs = model(img)
    prediction = outputs.argmax(dim=1).item()

print(f"Predicted Class: {labels[prediction]}")
