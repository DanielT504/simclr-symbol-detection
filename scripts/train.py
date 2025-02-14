import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

# transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),  # Randomly rotate by ±10 degrees
    transforms.RandomHorizontalFlip(),  # 50% chance to flip horizontally
    transforms.RandomVerticalFlip(),  # 50% chance to flip vertically
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class FirefightingDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None, num_classes=41):  # Adjusted num_classes to 41
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load label
        label_file = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                label_data = f.readlines()
                labels = [int(line.strip().split()[0]) for line in label_data]

                if len(labels) == 0:
                    label = 0  # Default to background class if no labels
                else:
                    label = max(set(labels), key=labels.count)  # Choose most common class
        else:
            label = 0  # If no label file exists, assign to background class

        if self.transform:
            image = self.transform(image)

        return image, label

# Load dataset
train_dataset = FirefightingDataset("dataset/train/images", "dataset/train/labels", transform=transform, num_classes=41)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

valid_dataset = FirefightingDataset("dataset/valid/images", "dataset/valid/labels", transform=transform, num_classes=41)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Define model
class SimCLRClassifier(nn.Module):
    def __init__(self, num_classes=41):
        super(SimCLRClassifier, self).__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.5)  # Dropout to reduce overfitting
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        output = self.classifier(features)
        return output

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRClassifier(num_classes=41).to(device)

# Compute class weights to balance dataset
class_counts = {
    0: 940, 3: 136, 4: 133, 5: 58, 16: 103, 24: 86, 21: 151, 33: 33, 1: 93, 40: 18, 32: 18, 22: 39, 
    28: 25, 8: 23, 7: 31, 25: 60, 27: 35, 34: 25, 15: 341, 23: 30, 2: 16, 17: 14, 38: 13, 26: 8, 
    19: 7, 9: 7, 37: 7, 31: 7, 11: 4, 29: 4, 14: 15, 39: 14, 10: 4, 13: 34, 36: 16, 20: 12, 
    30: 7, 6: 2, 18: 30, 35: 7
}

total_samples = sum(class_counts.values())
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
weights = torch.tensor([class_weights.get(i, 1.0) for i in range(41)]).to(device)

# Define loss function with class balancing
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    val_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()  # Ensure labels are LongTensor

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validate after each epoch
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(valid_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "models/simclr_firefighting.pth")
print("Training complete. Model saved!")
