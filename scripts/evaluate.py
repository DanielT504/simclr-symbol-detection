import torch
from train import SimCLRClassifier, transform, FirefightingDataset
from torch.utils.data import DataLoader

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load("models/simclr_firefighting.pth"))
model.eval()

# Load test data
test_dataset = FirefightingDataset("dataset/test/images", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
