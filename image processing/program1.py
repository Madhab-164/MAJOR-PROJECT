import os
import cv2
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from skimage.feature import hog
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import time
from tqdm import tqdm

class LungCancerClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_labels = {
            "Malignant cases": 0,
            "Normal cases": 1,
            "Benign cases": 2
        }
        self.models = {
            'svm': None,
            'rf': None
        }
        self.scaler = StandardScaler()
        self.pca = None
        self.best_params = {}
        self.results = {}
        
    def load_images_from_folder(self):
        """Load and preprocess images from folder structure"""
        X = []
        y = []
        missing_folders = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        print("\n📂 Loading dataset...")
        
        for class_name, label in self.class_labels.items():
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                missing_folders.append(class_name)
                continue
                
            print(f"Processing {class_name}...")
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(valid_extensions)]
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    features = self.process_image(img_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"\n⚠ Error processing {img_file}: {str(e)}")
                    continue
        
        if not X:
            raise ValueError("❌ No images loaded! Check your dataset path and structure.")
            
        print(f"\n✅ Successfully loaded {len(X)} images")
        if missing_folders:
            print(f"⚠ Missing folders: {', '.join(missing_folders)}")
            
        return np.array(X), np.array(y)
        
    def process_image(self, img_path):
        """Enhanced image preprocessing pipeline"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {img_path}")
            
        # Resize with aspect ratio preservation
        img = cv2.resize(img, (128, 128))
        
        # Adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction
        img = cv2.medianBlur(img, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Feature extraction
        features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), feature_vector=True)
        
        return features
        
    def handle_class_imbalance(self, X, y):
        """Apply SMOTE to balance class distribution"""
        print("\n⚖ Checking class distribution...")
        class_counts = Counter(y)
        print("Original distribution:", dict(class_counts))
        
        if len(set(class_counts.values())) > 1:
            print("Applying SMOTE to balance classes...")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            print("New distribution:", dict(Counter(y)))
            
        return X, y
        
    def train_models(self, X_train, y_train):
        """Train models with hyperparameter tuning"""
        print("\n🎯 Training models...")
        
        # SVM with Grid Search
        print("\n🔍 Tuning SVM...")
        svm_param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [1, 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42),
            svm_param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        svm_grid.fit(X_train, y_train)
        self.models['svm'] = svm_grid.best_estimator_
        self.best_params['svm'] = svm_grid.best_params_
        print(f"✅ Best SVM parameters: {svm_grid.best_params_}")
        
        # Random Forest with Grid Search
        print("\n🔍 Tuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        rf_grid.fit(X_train, y_train)
        self.models['rf'] = rf_grid.best_estimator_
        self.best_params['rf'] = rf_grid.best_params_
        print(f"✅ Best RF parameters: {rf_grid.best_params_}")
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate and compare model performance"""
        print("\n📊 Evaluating models...")
        
        for model_name, model in self.models.items():
            if model is None:
                print(f"⚠ {model_name} model not trained!")
                continue
                
            print(f"\nEvaluating {model_name}...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            self.results[model_name] = metrics
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=self.class_labels.keys()))
            
            # Confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_labels.keys(),
                   yticklabels=self.class_labels.keys())
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png')
        plt.show()
        
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("\n💾 Saving models...")
        os.makedirs('saved_models', exist_ok=True)
        
        for model_name, model in self.models.items():
            if model:
                joblib.dump(model, f'saved_models/{model_name}_model.pkl')
        
        joblib.dump(self.scaler, 'saved_models/scaler.pkl')
        if self.pca:
            joblib.dump(self.pca, 'saved_models/pca.pkl')
            
        print("✅ Models saved in 'saved_models' directory")
        
    def run(self):
        """Main execution pipeline"""
        try:
            # 1. Load and preprocess data
            X, y = self.load_images_from_folder()
            
            # 2. Handle class imbalance
            X, y = self.handle_class_imbalance(X, y)
            
            # 3. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
                
            # 4. Feature scaling
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # 5. Train models
            self.train_models(X_train, y_train)
            
            # 6. Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # 7. Save models
            self.save_models()
            
            return {
                'results': self.results,
                'best_params': self.best_params
            }
            
        except Exception as e:
            print(f"\n❌ Error encountered: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Verify dataset path exists and is accessible")
            print("2. Check folder structure matches expected format")
            print("3. Ensure images are in supported formats (jpg, png, etc.)")
            print("4. Confirm all required folders (Malignant, Normal, Benign) are present")
            return None


if __name__ == "__main__":
    # Initialize with your dataset path
    dataset_path = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Test Case"
    
    # Verify path exists
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        print("Current working directory:", os.getcwd())
    else:
        print(f"\n🏁 Starting Lung Cancer Classification System")
        print(f"Dataset path: {dataset_path}")
        
        # Run the system
        classifier = LungCancerClassifier(dataset_path)
        results = classifier.run()
        
        # Display results if successful
        if results:
            print("\n🎉 Final Results:")
            for model_name, metrics in results['results'].items():
                print(f"\n{model_name.upper()} Performance:")
                for metric, value in metrics.items():
                    print(f"{metric.capitalize()}: {value:.4f}")
            
            print("\n🔧 Best Parameters Found:")
            for model_name, params in results['best_params'].items():
                print(f"\n{model_name.upper()}:")
                for param, value in params.items():
                    print(f"{param}: {value}")