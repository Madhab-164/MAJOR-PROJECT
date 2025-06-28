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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tkinter import Tk, ttk, messagebox, filedialog
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
import imageio
from skimage.filters import threshold_otsu, unsharp_mask
from skimage import exposure
import pywt
import tkinter as tk
from tkinter import ttk

# Configuration
CONFIG = {
    'MODEL_PATHS': {
        'rf': 'models/rf_model.pkl',
        'cnn': 'models/cnn_attn_model.pth',
        'pca': 'models/pca.pkl'
    },
    'DATA_PATHS': {
        'train': {
            'Malignant': r"E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Train Case/Malignant cases",
            'Normal': r"E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Train Case/Normal cases",
            'Benign': r"E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Train Case/Benign cases"
        },
        'test': r"E:/MAJOR PROJECT/LUNG_PROJECT/New Dataset/Using Case"
    },
    'CLASSES': ['Malignant', 'Normal', 'Benign'],
    'IMG_SIZE': (512, 512),
    'SEED': 42,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 1e-5,
    'EPOCHS': 30
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Enhanced Medical Image Processor ------------------------
class MedicalImageProcessor:
    def __init__(self):
        self.filters = {
            'Gaussian Blur': lambda img: cv2.GaussianBlur(img, (9, 9), 2),
            'Median Blur': lambda img: cv2.medianBlur(img, 7),
            'Bilateral Filter': lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            'Sobel X': self._sobel_x,
            'Sobel Y': self._sobel_y,
            'Laplacian': self._laplacian,
            'Canny Edge': self._canny_edge,
            'Unsharp Mask': self._unsharp_mask,
            'Wavelet Transform': self._wavelet_transform,
            'Guided Filter': self._guided_filter,
            'Adaptive Threshold': self._adaptive_threshold,
            'Histogram Equalization': self._equalize_histogram
        }

    def process_image(self, img, filter_name):
        try:
            if isinstance(img, Image.Image):
                img = np.array(img)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img = self._convert_to_uint8(img)
            
            if filter_name not in self.filters:
                raise ValueError(f"Unknown filter: {filter_name}")
                
            return self.filters[filter_name](img)
        except Exception as e:
            raise RuntimeError(f"Error in {filter_name}: {str(e)}")

    def _convert_to_uint8(self, img):
        if img.dtype == np.float64:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return img.astype(np.uint8)
        elif img.dtype == np.float32:
            return (img * 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            return (img // 256).astype(np.uint8)
        return img.astype(np.uint8)

    def _sobel_x(self, img):
        img = self._ensure_grayscale(img)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        return self._convert_to_uint8(np.absolute(sobelx))

    def _sobel_y(self, img):
        img = self._ensure_grayscale(img)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        return self._convert_to_uint8(np.absolute(sobely))

    def _laplacian(self, img):
        img = self._ensure_grayscale(img)
        laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
        return self._convert_to_uint8(np.absolute(laplacian))

    def _canny_edge(self, img):
        img = self._ensure_grayscale(img)
        return cv2.Canny(img, 100, 200)

    def _unsharp_mask(self, img):
        img = self._ensure_grayscale(img)
        img_float = img.astype(np.float32) / 255.0
        sharpened = unsharp_mask(img_float, radius=5, amount=2)
        return self._convert_to_uint8(sharpened)

    def _wavelet_transform(self, img):
        img = self._ensure_grayscale(img)
        coeffs = pywt.dwt2(img, 'haar')
        reconstructed = pywt.idwt2(coeffs, 'haar')
        return self._convert_to_uint8(reconstructed)

    def _guided_filter(self, img, radius=15, eps=0.01):
        img = self._ensure_grayscale(img)
        return cv2.ximgproc.guidedFilter(img, img, radius, eps)

    def _adaptive_threshold(self, img):
        img = self._ensure_grayscale(img)
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    def _equalize_histogram(self, img):
        img = self._ensure_grayscale(img)
        equalized = exposure.equalize_adapthist(img)
        return self._convert_to_uint8(equalized)

    def _ensure_grayscale(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

# ------------------------ CNN Model ------------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H*W)
        energy = torch.bmm(query, key)
        attention = torch.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class CancerResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        layers = list(self.base_model.children())[:-2]
        self.feature_extractor = nn.Sequential(*layers)
        self.attention = AttentionBlock(2048)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(CONFIG['CLASSES']))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.attention(x)
        return self.classifier(x)

# ------------------------ GUI Application ------------------------
class CancerDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.processor = MedicalImageProcessor()
        self.initialize_models()
        self.setup_gui()

    def initialize_models(self):
        self.rf_model = RandomForestClassifier()
        self.cnn_model = CancerResNet().to(device)
        
        try:
            with open(CONFIG['MODEL_PATHS']['rf'], 'rb') as f:
                self.rf_model = pickle.load(f)
            
            cnn_state = torch.load(CONFIG['MODEL_PATHS']['cnn'], map_location=device)
            self.cnn_model.load_state_dict(cnn_state)
            self.cnn_model.eval()
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load models: {str(e)}")
            self.root.destroy()

    def setup_gui(self):
        self.root.title("Lung Cancer Diagnosis System")
        self.root.geometry("1200x800")
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#ffffff')
        self.style.configure('TLabel', background='#ffffff', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12), padding=5)

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=10)

        self.control_frame = ttk.Frame(self.main_frame, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.load_button = ttk.Button(
            self.control_frame,
            text="Load CT Scan",
            command=self.load_image
        )
        self.load_button.pack(pady=10, fill=tk.X)

        self.process_button = ttk.Button(
            self.control_frame,
            text="Process Image",
            command=self.process_image,
            state=tk.DISABLED
        )
        self.process_button.pack(pady=10, fill=tk.X)

        self.analyze_button = ttk.Button(
            self.control_frame,
            text="Analyze for Cancer",
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.analyze_button.pack(pady=10, fill=tk.X)

        self.results_frame = ttk.LabelFrame(self.control_frame, text="Analysis Results")
        self.results_frame.pack(pady=20, fill=tk.X)
        self.results_text = tk.Text(self.results_frame, height=10, width=30, state=tk.DISABLED)
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH)

        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.tiff")]
        )
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.current_image.thumbnail((600, 600))
                img_tk = ImageTk.PhotoImage(self.current_image)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.process_button.config(state=tk.NORMAL)
                self.analyze_button.config(state=tk.DISABLED)
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load image: {str(e)}")

    def process_image(self):
        try:
            cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            processed = self.processor.process_image(cv_image, 'Gaussian Blur')
            processed = self.processor.process_image(processed, 'Histogram Equalization')
            
            if processed.ndim == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                
            self.processed_image = Image.fromarray(processed)
            display_image = self.processed_image.copy()
            display_image.thumbnail((600, 600))
            img_tk = ImageTk.PhotoImage(display_image)
            
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
            self.analyze_button.config(state=tk.NORMAL)
            self.status_var.set("Image processed successfully")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Image processing failed: {str(e)}")

    def analyze_image(self):
        try:
            img_tensor = self._prepare_image_for_model(self.processed_image)
            rf_input = self._prepare_image_for_rf(self.processed_image)

            with torch.no_grad():
                cnn_output = self.cnn_model(img_tensor)
                cnn_probs = torch.nn.functional.softmax(cnn_output, dim=1)
                cnn_pred = torch.argmax(cnn_probs, dim=1).item()

            rf_pred = self.rf_model.predict(rf_input)
            self._display_results(cnn_pred, rf_pred, cnn_probs)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze image: {str(e)}")

    def _prepare_image_for_model(self, image):
        transform = transforms.Compose([
            transforms.Resize(CONFIG['IMG_SIZE']),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(device)

    def _prepare_image_for_rf(self, image):
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, CONFIG['IMG_SIZE'])
        return resized.reshape(1, -1)

    def _display_results(self, cnn_pred, rf_pred, cnn_probs):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        classes = CONFIG['CLASSES']
        self.results_text.insert(tk.END, "CNN Analysis:\n")
        self.results_text.insert(tk.END, f"Prediction: {classes[cnn_pred]}\n")
        for i, prob in enumerate(cnn_probs[0]):
            self.results_text.insert(tk.END, f"{classes[i]}: {prob:.2%}\n")
        
        self.results_text.insert(tk.END, "\nRandom Forest Analysis:\n")
        self.results_text.insert(tk.END, f"Prediction: {classes[rf_pred[0]]}\n")
        self.results_text.config(state=tk.DISABLED)
        self.status_var.set("Analysis complete")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    root = Tk()
    app = CancerDiagnosisApp(root)
    root.mainloop()