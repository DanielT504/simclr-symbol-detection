import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(10),  # Randomly rotate by ±10 degrees
    transforms.RandomHorizontalFlip(),  # 50% chance to flip
    transforms.RandomVerticalFlip(),  # 50% chance to flip vertically
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness & contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Custom dataset loader
class FirefightingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = 0  # Placeholder (change based on labels in dataset)

        if self.transform:
            image = self.transform(image)

        return image, label

# Load dataset
train_dataset = FirefightingDataset("dataset/train/images", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model
class SimCLRClassifier(nn.Module):
    def __init__(self, num_classes=5):
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
model = SimCLRClassifier(num_classes=5).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
valid_dataset = FirefightingDataset("dataset/valid/images", transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    val_loss = 0.0


    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(valid_loader)}")


# Save the model
torch.save(model.state_dict(), "models/simclr_firefighting.pth")
print("Training complete. Model saved!")
