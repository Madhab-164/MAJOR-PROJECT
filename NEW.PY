import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename
from scipy.ndimage import gaussian_filter
import imageio
from PIL import Image
from skimage.filters import threshold_otsu  # For Otsu's thresholding

from eval import LungCancerDataset

# Configuration
CONFIG = {
    'MODEL_PATHS': {
        'svm': 'models/svm_model.pkl',
        'cnn': 'models/cnn_model.pth',
        'pca': 'models/pca.pkl'
    },
    'DATA_PATHS': {
        'train': "Dataset/Data/train",
        'test': "Dataset/Data/test"
    },
    'CLASSES': {
        'Normal_Cell': 0,
        'Adenocarcinoma': 1,
        'Large_Cell_Carcinoma': 2,
        'Squamous_Cell_Carcinoma': 3
    },
    'IMG_SIZE': (224, 224),
    'SEED': 42
}

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Enhanced Data Loading ------------------------
class DataHandler:
    def __init__(self):
        self.classes = CONFIG['CLASSES']
        self.img_size = CONFIG['IMG_SIZE']
        
    def load_dataset(self, dataset_path, color_mode='rgb'):
        images = []
        labels = []
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        for class_name, class_id in self.classes.items():
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                raise FileNotFoundError(f"Class directory {class_path} not found")

            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                try:
                    img = self._load_image(img_path, color_mode)
                    images.append(img)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels)

    def _load_image(self, path, color_mode):
        if color_mode == 'grayscale':
            img = imageio.imread(path, as_gray=True)
            img = (img * 255).astype(np.uint8)  # Convert to 8-bit
        else:
            img = imageio.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
            
        img = cv2.resize(img, self.img_size)
        
        if color_mode == 'rgb':
            img = img.astype(np.float32) / 255.0
            
        return img

# ------------------------ Model Management ------------------------
class ModelManager:
    def __init__(self):
        self.config = CONFIG
        os.makedirs('models', exist_ok=True)

    def train_or_load_svm(self):
        if os.path.exists(self.config['MODEL_PATHS']['svm']):
            return self._load_models()
            
        print("Training SVM model...")
        dh = DataHandler()
        X_train, y_train = dh.load_dataset(self.config['DATA_PATHS']['train'], 'grayscale')
        X_test, y_test = dh.load_dataset(self.config['DATA_PATHS']['test'], 'grayscale')

        # Flatten and normalize
        X_train_flat = X_train.reshape(len(X_train), -1) / 255.0
        X_test_flat = X_test.reshape(len(X_test), -1) / 255.0

        # Dimensionality reduction
        pca = PCA(n_components=0.98)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_test_pca = pca.transform(X_test_flat)

        # Model training
        svm_model = SVC(kernel='rbf', C=100, random_state=CONFIG['SEED'])
        svm_model.fit(X_train_pca, y_train)

        # Save models
        self._save_models(svm_model, pca)
        return svm_model, pca

    def train_or_load_cnn(self):
        if os.path.exists(self.config['MODEL_PATHS']['cnn']):
            return self._load_cnn()
            
        print("Training CNN model...")
        dh = DataHandler()
        X_train, y_train = dh.load_dataset(self.config['DATA_PATHS']['train'], 'rgb')
        X_test, y_test = dh.load_dataset(self.config['DATA_PATHS']['test'], 'rgb')

        # Create datasets
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(CONFIG['IMG_SIZE']),
            transforms.ToTensor()
        ])
        
        train_dataset = LungCancerDataset(X_train, y_train, transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Model setup
        model = CNNModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training
        self._train_cnn(model, train_loader, criterion, optimizer)
        torch.save(model.state_dict(), self.config['MODEL_PATHS']['cnn'])
        return model

    def _save_models(self, svm_model, pca):
        with open(CONFIG['MODEL_PATHS']['svm'], 'wb') as f:
            pickle.dump({'model': svm_model, 'pca': pca}, f)
        print("SVM model and PCA saved")

    def _load_models(self):
        try:
            with open(CONFIG['MODEL_PATHS']['svm'], 'rb') as f:
                data = pickle.load(f)
            print("Loaded existing SVM model and PCA")
            return data['model'], data['pca']
        except Exception as e:
            print(f"Error loading SVM model: {str(e)}")
            raise

    def _load_cnn(self):
        try:
            model = CNNModel().to(device)
            model.load_state_dict(torch.load(CONFIG['MODEL_PATHS']['cnn']))
            model.eval()
            print("Loaded existing CNN model")
            return model
        except Exception as e:
            print(f"Error loading CNN model: {str(e)}")
            raise

    def _train_cnn(self, model, loader, criterion, optimizer, epochs=10):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

# ------------------------ CNN Architecture ------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, len(CONFIG['CLASSES']))

    def forward(self, x):
        return self.base_model(x)

# ------------------------ Enhanced Image Processing Pipeline ------------------------
class ImageAnalyzer:
    def __init__(self, svm_model, pca, cnn_model):
        self.svm_model = svm_model
        self.pca = pca
        self.cnn_model = cnn_model
        self.class_names = list(CONFIG['CLASSES'].keys())

    def analyze_image(self, img_path):
        try:
            # Load and verify image
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            original_img = imageio.imread(img_path)
            if original_img is None:
                raise ValueError("Invalid image file")

            # Processing pipeline
            self._display_image(original_img, "Original Image")
            processed = self._preprocessing_pipeline(original_img)
            predictions = self._make_predictions(processed)
            self._show_results(original_img, predictions)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

    def _preprocessing_pipeline(self, img):
        # Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, CONFIG['IMG_SIZE'])
        
        # Noise reduction using Gaussian filter
        filtered = gaussian_filter(resized, sigma=1)
        
        # Segmentation using Otsu's method (from scikit-image)
        thresh = threshold_otsu(filtered)
        binary = filtered > thresh
        
        return {
            'gray': gray,
            'resized': resized,
            'filtered': filtered,
            'binary': binary
        }

    def _make_predictions(self, processed):
        # SVM Prediction
        svm_input = processed['resized'].astype(np.float32) / 255.0
        svm_input_pca = self.pca.transform(svm_input.reshape(1, -1))
        svm_pred = self.svm_model.predict(svm_input_pca)[0]

        # CNN Prediction
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(CONFIG['IMG_SIZE']),
            transforms.ToTensor()
        ])
        
        cnn_input = transform(cv2.resize(processed['gray'], CONFIG['IMG_SIZE'])).unsqueeze(0).to(device)
        with torch.no_grad():
            cnn_output = self.cnn_model(cnn_input)
            cnn_pred = torch.argmax(cnn_output, 1).item()

        return {
            'svm': self.class_names[svm_pred],
            'cnn': self.class_names[cnn_pred]
        }

    def _display_image(self, img, title):
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def _show_results(self, original_img, predictions):
        result_img = cv2.resize(original_img, CONFIG['IMG_SIZE'])
        text = f"SVM: {predictions['svm']} | CNN: {predictions['cnn']}"
        cv2.putText(result_img, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self._display_image(result_img, "Diagnosis Result")

# ------------------------ Main Application Flow ------------------------
def main():
    # Initialize model manager
    mm = ModelManager()
    
    try:
        # Load or train models
        svm_model, pca = mm.train_or_load_svm()
        cnn_model = mm.train_or_load_cnn()
        
        # Initialize analyzer
        analyzer = ImageAnalyzer(svm_model, pca, cnn_model)
        
        # File selection dialog
        Tk().withdraw()
        img_path = askopenfilename(title="Select Lung Image", 
                                 filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not img_path:
            print("No image selected")
            return
            
        # Process selected image
        analyzer.analyze_image(img_path)
        
    except Exception as e:
        messagebox.showerror("Application Error", str(e))
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()