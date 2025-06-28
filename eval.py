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
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Set global parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 14
sns.set_style('whitegrid')

# ------------------------ Constants ------------------------
TRAIN_PATH = "E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Train Case"
TEST_PATH = "E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Test Case"

CLASSES = {
    'Malignant cases': 0,
    'Normal cases': 1,
    'Benign cases': 2
}
IMG_SIZE = (224, 224)

# ------------------------ Data Loading & Preprocessing ------------------------
def load_data(dataset_path, classes, img_size=IMG_SIZE, color_mode='grayscale'):
    images = []
    labels = []
    
    for class_name, class_id in classes.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            raise FileNotFoundError(f"Directory not found: {class_path}")
            
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
    
    return np.array(images), np.array(labels)

# Load SVM data (grayscale)
X_train_svm, y_train_svm = load_data(TRAIN_PATH, CLASSES, color_mode='grayscale')
X_test_svm, y_test_svm = load_data(TEST_PATH, CLASSES, color_mode='grayscale')

# Load CNN data (RGB)
X_train_cnn, y_train_cnn = load_data(TRAIN_PATH, CLASSES, color_mode='rgb')
X_test_cnn, y_test_cnn = load_data(TEST_PATH, CLASSES, color_mode='rgb')

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
    transforms.Resize(IMG_SIZE),
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

# Training Loop
def train_model(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

train_model(cnn_model, train_loader)

# ------------------------ Model Saving ------------------------
torch.save(cnn_model.state_dict(), 'cnn_lung_cancer_detector.pth')
import pickle
with open('svm_lung_cancer_detector.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

print("All models trained and saved successfully!")

# ------------------------ Image Upload & Processing Pipeline ------------------------
def process_uploaded_image(IMG_SIZE, pca, svm_model, device, cnn_model, CLASSES):
    # Image upload dialog
    Tk().withdraw()
    img_path = askopenfilename(title="Select Lung Image")
    
    if not img_path:
        print("No image selected")
        return

    # Load and display original image
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.title("Original Uploaded Image")
    plt.axis('off')
    plt.show()

    # ----------------- Preprocessing -----------------
    # Convert to grayscale and resize
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(gray_img, IMG_SIZE)
    
    # Normalization
    normalized_img = resized_img.astype(np.float32) / 255.0

    # ----------------- Noise Analysis -----------------
    laplacian = cv2.Laplacian(resized_img, cv2.CV_64F)
    noise_level = laplacian.var()
    print(f"Detected Noise Level: {noise_level:.2f}")

    # ----------------- Filtering -----------------
    # Apply median filtering
    filtered_img = cv2.medianBlur(resized_img, 5)
    
    # Apply bilateral filtering
    bilateral_img = cv2.bilateralFilter(resized_img, 9, 75, 75)

    # ----------------- Segmentation -----------------
    # Thresholding
    _, thresh_img = cv2.threshold(filtered_img, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    # Create segmentation mask
    segmented_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)

    # ----------------- Visualization -----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original Image
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Preprocessed Image
    axs[0, 1].imshow(resized_img, cmap='gray')
    axs[0, 1].set_title('Preprocessed Image')
    axs[0, 1].axis('off')
    
    # Filtered Images
    axs[0, 2].imshow(filtered_img, cmap='gray')
    axs[0, 2].set_title('Median Filtered')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(bilateral_img, cmap='gray')
    axs[1, 0].set_title('Bilateral Filtered')
    axs[1, 0].axis('off')
    
    # Thresholded Image
    axs[1, 1].imshow(thresh_img, cmap='gray')
    axs[1, 1].set_title('Thresholded Image')
    axs[1, 1].axis('off')
    
    # Segmented Image
    axs[1, 2].imshow(segmented_img)
    axs[1, 2].set_title('Segmented Image with Contours')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # ----------------- Classification -----------------
    # SVM Prediction
    svm_input = normalized_img.reshape(1, -1)
    svm_input_pca = pca.transform(svm_input)
    svm_pred = svm_model.predict(svm_input_pca)[0]
    
    # CNN Prediction
    cnn_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])(cv2.resize(original_img, IMG_SIZE)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cnn_output = cnn_model(cnn_input)
        cnn_pred = torch.argmax(cnn_output, 1).item()
    
    class_names = list(CLASSES.keys())
    
    # ----------------- Display Final Result -----------------
    result_img = cv2.resize(original_img, IMG_SIZE)
    text = f"SVM: {class_names[svm_pred]} | CNN: {class_names[cnn_pred]}"
    cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(result_img)
    plt.title("Final Diagnosis Result")
    plt.axis('off')
    plt.show()

# Call the function with the required parameters
process_uploaded_image(IMG_SIZE, pca, svm_model, device, cnn_model, CLASSES)