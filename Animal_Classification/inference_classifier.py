import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from model import CNNModel

ANIMAL_NAMES = ["dog", "cat", "elephant", "lion", "tiger", "bear", "zebra", "cow", "sheep", "horse"]

MODEL_PATH = "models/cnn_model.pth"
model = CNNModel(num_classes=len(ANIMAL_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_animal(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return ANIMAL_NAMES[predicted.item()]
    except Exception:
        return None

