import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
data_path = './101_ObjectCategories'
caltech_dataset = datasets.ImageFolder(root=data_path, transform=transform)
classes = caltech_dataset.classes
num_classes = len(classes)

# Split dataset
train_size = int(0.7 * len(caltech_dataset))
val_size = int(0.2 * len(caltech_dataset))
test_size = len(caltech_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(caltech_dataset, [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained model and modify for fine-tuning
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Plot learning curves
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set and visualize metrics
model.eval()
test_predictions = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_predictions.extend(predicted.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

report = classification_report(test_labels, test_predictions, digits=4)
print(report)

conf_matrix = confusion_matrix(test_labels, test_predictions)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
test_labels_array = np.array(test_labels)
test_predictions_array = np.array(test_predictions)
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve((test_labels_array == i).astype(int), (test_predictions_array == i).astype(int))
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], marker='o', markersize=4)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for each class')
plt.show()

precision = dict()
recall = dict()
prc_auc = dict()
for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve((test_labels_array == i).astype(int), (test_predictions_array == i).astype(int))
    prc_auc[i] = auc(recall[i], precision[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(recall[i], precision[i], marker='o', markersize=4, lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for each class')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.show()

total_samples = len(test_labels)
incorrect_predictions = total_samples - np.sum(test_predictions_array == test_labels_array)
top1_error_rate = (incorrect_predictions / total_samples) * 100
print("Top-1 Error Rate: {:.2f}%".format(top1_error_rate))

# Visualize filters for the first convolutional layer
filters = model.conv1.weight.data.clone()
filters = filters - filters.min()
filters = filters / filters.max()
num_filters = filters.shape[0]
num_channels = filters.shape[1]
filter_height = filters.shape[2]
filter_width = filters.shape[3]
plt.figure(figsize=(16, 12))
for i in range(num_filters):
    for j in range(num_channels):
        plt.subplot(num_filters, num_channels, i * num_channels + j + 1)
        plt.imshow(filters[i, j], cmap='gray')
        plt.axis('off')
plt.show()

# Visualize feature maps after the first convolutional layer
input_img = torch.randn(1, 3, 64, 64).to(device)
output = model.conv1(input_img)
feature_maps = output.squeeze().detach().cpu().numpy()
num_feature_maps = feature_maps.shape[0]
plt.figure(figsize=(16, 12))
for i in range(num_feature_maps):
    plt.subplot(num_feature_maps // 8 + 1, 8, i + 1)
    plt.imshow(feature_maps[i], cmap='viridis')
    plt.axis('off')
plt.show()

torch.save(model.state_dict(), 'frozen.pth')
