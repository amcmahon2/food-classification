import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import ImageFile
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

#delete empty or invalid folders
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
split_dirs = ['../data/split/train', '../data/split/val']
for split_dir in split_dirs:
    for folder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if not file.lower().endswith(VALID_EXTENSIONS):
                    print(f"Deleting non-image file: {file_path}")
                    os.remove(file_path)
            if not any(fname.lower().endswith(VALID_EXTENSIONS) for fname in os.listdir(folder_path)):
                print(f"Removing empty folder: {folder_path}")
                os.rmdir(folder_path)

#hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 0.001
DATA_DIR = '../data/split'

#data augmentation and transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

#load datasets
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform)
with open("../models/classes.txt", "w") as f:
    for cls in train_dataset.classes:
        f.write(cls + "\n")
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

#load pretrained model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

#loss and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100 * correct / total
    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    scheduler.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%")

#save model
torch.save(model.state_dict(), "food_classifier.pth")
print("Model saved to food_classifier.pth")