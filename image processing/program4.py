import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, RocCurveDisplay, precision_recall_curve)
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycaret.classification import *
from io import StringIO
import sys
from itertools import cycle

class Config:
    DATA_DIR = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Train Case"
    CLASSES = ['Benign cases', 'Normal cases', 'Malignant cases']
    NUM_FOLDS = 5
    IMG_SIZE = (224, 224)
    RANDOM_SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    METRICS_DIR = "./metrics_plots"

np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)

# Create metrics directory
os.makedirs(Config.METRICS_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(Config.IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def plot_confusion_matrix(y_true, y_pred, fold, model_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=Config.CLASSES, yticklabels=Config.CLASSES)
    plt.title(f'{model_type} Confusion Matrix (Fold {fold})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(Config.METRICS_DIR, f'{model_type}_confusion_matrix_fold_{fold}.png'))
    plt.close()

def plot_roc_curve(y_true, y_score, fold, model_type):
    n_classes = len(Config.CLASSES)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {Config.CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_type} ROC Curve (Fold {fold})')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(Config.METRICS_DIR, f'{model_type}_roc_curve_fold_{fold}.png'))
    plt.close()

def plot_auc_curves(y_true, y_score, fold, model_type):
    n_classes = len(Config.CLASSES)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    precision = dict()
    recall = dict()
    auc_score = dict()
    
    plt.figure(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        auc_score[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve {Config.CLASSES[i]} (AUC = {auc_score[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{model_type} Precision-Recall Curve (Fold {fold})')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(Config.METRICS_DIR, f'{model_type}_pr_curve_fold_{fold}.png'))
    plt.close()

def evaluate_metrics(y_true, y_pred, y_score, fold, model_type):
    print(f"\n{classification_report(y_true, y_pred, target_names=Config.CLASSES)}")
    
    plot_confusion_matrix(y_true, y_pred, fold, model_type)
    plot_roc_curve(y_true, y_score, fold, model_type)
    plot_auc_curves(y_true, y_score, fold, model_type)

class MedicalImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(MedicalCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, fold):
    best_acc = 0.0
    model = model.to(Config.DEVICE)
    
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        val_loss, val_acc, val_preds, val_probs = evaluate_pytorch_model(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
        print(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_pytorch_model_fold_{fold}.pth')
            print("Saved new best model!")
    
    return best_acc, val_preds, val_probs

def evaluate_pytorch_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(data_loader.dataset)
    acc = running_corrects.double() / len(data_loader.dataset)
    return loss, acc, np.array(all_preds), np.array(all_probs)

from pycaret.classification import *
import pandas as pd

def run_pycaret_experiment(train_data, test_data, fold):
    print(f"\n========================================")
    print(f"📌 Running PyCaret Experiment: Fold {fold}")
    print(f"========================================")

    # Step 1: Prepare DataFrames with Column Names
    feature_cols = [f'pixel_{i}' for i in range(train_data.shape[1] - 1)]
    train_df = pd.DataFrame(train_data.iloc[:, :-1].values, columns=feature_cols)
    train_df['label'] = train_data['label'].values

    test_df = pd.DataFrame(test_data.iloc[:, :-1].values, columns=feature_cols)
    test_df['label'] = test_data['label'].values

    try:
        # Step 2: Setup PyCaret
        clf = setup(
            data=train_df,
            target='label',
            session_id=42,
            normalize=True,
            transformation=False,
            fold_strategy='kfold',
            fold=3,
            verbose=False,
            html=False,
            log_experiment=False,  
            use_gpu=False,  # GPU not used here; adjust if needed
            experiment_name=f'lung_cancer_fold_{fold}'
        )

        # Step 3: Dynamically fetch only available models
        all_models = models()
        available_model_ids = [
        'lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge',
        'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm',
        'catboost', 'dummy'
        ]

        
        print(f"🧠 {len(available_model_ids)} models found: {available_model_ids}")

        # Step 4: Compare all available models
        best_model = compare_models(include=available_model_ids, sort='Accuracy', verbose=True)

        # Step 5: Tune Best Model
        tuned_model = tune_model(best_model, optimize='Accuracy', verbose=True)

        # Step 6: Finalize Model
        final_model = finalize_model(tuned_model)

        # Step 7: Predict on Test Data
        predictions = predict_model(final_model, data=test_df)
        pycaret_probs = predict_model(final_model, data=test_df, raw_score=True)

        # Step 8: Extract Scores and Metrics
        prob_columns = [col for col in pycaret_probs.columns if 'Score_' in col]
        y_score = pycaret_probs[prob_columns].values

        y_true = predictions['label']
        y_pred = predictions['prediction_label']
        accuracy = (y_true == y_pred).mean()

        # Step 9: Evaluate (custom function)
        evaluate_metrics(y_true, y_pred, y_score, fold, "PyCaret")

        # Step 10: Save Final Model
        save_model(final_model, f'pycaret_best_model_fold_{fold}')

        print(f"✅ Fold {fold} complete | Accuracy: {accuracy:.4f}")
        return accuracy, final_model, y_true, y_pred, y_score

    except Exception as e:
        print(f"❌ PyCaret experiment failed at Fold {fold}: {str(e)}")
        return 0.0, None, None, None, None

class redirect_output:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def load_and_preprocess_data():
    X = []
    y = []
    
    for idx, class_name in enumerate(Config.CLASSES):
        class_dir = os.path.join(Config.DATA_DIR, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
                X.append(image.numpy())
                y.append(idx)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} images.")
    return X, y


def main():
    try: 
        X, y = load_and_preprocess_data()
        kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.RANDOM_SEED)
        
        pycaret_metrics = []
        pytorch_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\n{'='*40}")
            print(f"Processing Fold {fold+1}/{Config.NUM_FOLDS}")
            print(f"{'='*40}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # PyCaret Experiment
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            train_data = pd.DataFrame(X_train_flat)
            train_data['label'] = y_train
            test_data = pd.DataFrame(X_test_flat)
            test_data['label'] = y_test
            
            pycaret_acc, _, y_true_pycaret, y_pred_pycaret, y_score_pycaret = run_pycaret_experiment(train_data, test_data, fold+1)
            pycaret_metrics.append(pycaret_acc)
            
            # PyTorch Experiment
            print("\nRunning PyTorch CNN")
            train_dataset = MedicalImageDataset(X_train, y_train)
            test_dataset = MedicalImageDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            model = MedicalCNN(num_classes=len(Config.CLASSES))
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            
            pytorch_acc, y_pred_pytorch, y_score_pytorch = train_pytorch_model(model, train_loader, test_loader, criterion, optimizer, fold+1)
            
            # Evaluate PyTorch metrics
            evaluate_metrics(y_test, y_pred_pytorch, y_score_pytorch, fold+1, "PyTorch")
            
            pytorch_metrics.append(pytorch_acc)
        
        # Final Results
        print(f"\n{'='*40}")
        print("Final Results:")
        print(f"{'='*40}")
        print(f"PyCaret Average Accuracy: {np.mean(pycaret_metrics):.4f} ± {np.std(pycaret_metrics):.4f}")
        print(f"PyTorch Average Accuracy: {np.mean(pytorch_metrics):.4f} ± {np.std(pytorch_metrics):.4f}")
        
        if np.mean(pycaret_metrics) > np.mean(pytorch_metrics):
            print("\nRecommendation: Use PyCaret with best traditional ML model")
        else:
            print("\nRecommendation: Use PyTorch CNN model")
    
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()