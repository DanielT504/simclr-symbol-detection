import torch
from train import SimCLRClassifier, transform, FirefightingDataset
from torch.utils.data import DataLoader

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimCLRClassifier(num_classes=41).to(device)  # Updated to 41 classes
model.load_state_dict(torch.load("models/simclr_firefighting.pth"))
model.eval()

# Load test data
test_dataset = FirefightingDataset("dataset/test/images", "dataset/test/labels", transform=transform, num_classes=41)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate
correct = 0
total = 0
class_correct = {i: 0 for i in range(41)}  # Track per-class accuracy
class_total = {i: 0 for i in range(41)}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device).long()  # Convert labels to LongTensor
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Track per-class accuracy
        for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

# Print overall accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Print per-class accuracy
print("\nPer-Class Accuracy:")
for cls in range(41):
    if class_total[cls] > 0:
        print(f"Class {cls}: {100 * class_correct[cls] / class_total[cls]:.2f}%")
    else:
        print(f"Class {cls}: No samples in test set")
