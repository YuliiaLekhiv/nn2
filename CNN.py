import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import os

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_path = './101_ObjectCategories'
caltech_dataset = datasets.ImageFolder(root=data_path, transform=transform)
classes = caltech_dataset.classes
num_classes = len(classes)

train_size = int(0.7 * len(caltech_dataset))
val_size = int(0.2 * len(caltech_dataset))
test_size = len(caltech_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(caltech_dataset,
                                                                         [train_size, val_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []
val_losses = []
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.2f}, Accuracy: {train_accuracy:.2f}%')

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy on the validation set: {(100 * correct / total):.2f}%')

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def collect_predictions_labels(model, dataset_loader):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataset_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    return all_predictions, all_labels


test_predictions, test_labels = collect_predictions_labels(model, test_loader)

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
    precision[i], recall[i], _ = precision_recall_curve((test_labels_array == i).astype(int),
                                                        (test_predictions_array == i).astype(int))
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


def visualize_filters(layer):
    filters = layer.weight.data.clone()
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


def visualize_feature_maps(model, layer_idx, input_img):
    output = None
    for idx, layer in enumerate(model.children()):
        input_img = layer(input_img)
        if idx == layer_idx:
            output = input_img
            break
    feature_maps = output.squeeze().detach().cpu().numpy()
    num_feature_maps = feature_maps.shape[0]
    plt.figure(figsize=(16, 12))
    for i in range(num_feature_maps):
        plt.subplot(num_feature_maps // 8 + 1, 8, i + 1)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')
    plt.show()


input_img = torch.randn(1, 3, 64, 64)
visualize_filters(model.conv1)
visualize_feature_maps(model, layer_idx=0, input_img=input_img)
visualize_filters(model.conv2)
visualize_feature_maps(model, layer_idx=1, input_img=input_img)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Learning curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

torch.save(model.state_dict(), f'3layercnn.pth')
