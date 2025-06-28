import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

# Set global parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
sns.set_style('whitegrid')

# ------------------------ Dataset Path ------------------------
DATASET_PATH = "E:/MAJOR PROJECT/LUNG_PROJECT/Dataset/Data/test"  # Single dataset path

# ------------------------ Data Loading & Preprocessing ------------------------
def load_data(dataset_path, img_size=(224, 224), color_mode='grayscale'):
    images = []
    labels = []
    classes = {}
    
    # Automatically detect classes from subdirectories
    for class_id, class_name in enumerate(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        classes[class_name] = class_id
        
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            try:
                if color_mode == 'grayscale':
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                
                img = cv2.resize(img, img_size)
                
                if color_mode == 'rgb':
                    img = img.astype(np.float32) / 255.0  # Normalize
                
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels), classes

# Load data
X, y, CLASSES = load_data(DATASET_PATH, color_mode='grayscale')
X_rgb, y_rgb, _ = load_data(DATASET_PATH, color_mode='rgb')

# Split data into train and test sets
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.5, random_state=42)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_rgb, y_rgb, test_size=0.5, random_state=42)

# ------------------------ SVM Implementation ------------------------
X_train_flat = X_train_svm.reshape(len(X_train_svm), -1) / 255.0
X_test_flat = X_test_svm.reshape(len(X_test_svm), -1) / 255.0

# PCA Implementation
pca = PCA(n_components=0.98)
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

svm_model = SVC(kernel='rbf', C=100, random_state=42)
svm_model.fit(X_train_pca, y_train_svm)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES.keys()))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES.keys(), yticklabels=CLASSES.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

evaluate_model(svm_model, X_train_pca, X_test_pca, y_train_svm, y_test_svm)

# ------------------------ CNN Implementation (PyTorch) ------------------------
class LungCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Creating Datasets
train_dataset = LungCancerDataset(X_train_cnn, y_train_cnn, transform=transform)
test_dataset = LungCancerDataset(X_test_cnn, y_test_cnn, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, len(CLASSES))

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = CNNModel().to(device)

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)

# Modified Training Loop (single epoch)
def train_model(model, train_loader):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Training Loss: {running_loss:.4f}")

train_model(cnn_model, train_loader)

# ------------------------ Model Saving ------------------------
torch.save(cnn_model.state_dict(), 'cnn_lung_cancer_detector.pth')
import pickle
with open('svm_lung_cancer_detector.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

print("All models trained and saved successfully!")