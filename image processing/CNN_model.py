import os
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
import torch.optim as optim

# ——— CONFIGURATION ————————————————————————————
DATA_DIR = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Train Case"
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_FOLDS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— QUANTUM-INSPIRED DROPOUT (Custom) —————————
class QuantumDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask1 = (torch.rand_like(x) > self.p).float()
        mask2 = 1.0 - mask1
        mask = mask1 if torch.rand(1).item() < 0.5 else mask2
        return x * mask / (1.0 - self.p)

# ——— CNN MODEL ————————————————————————————————
class LungCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            QuantumDropout(p=0.3),
            nn.Linear(base.fc.in_features, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

# ——— MAIN TRAINING FUNCTION ————————————————————————
def run_training():
    # AUGMENTATION PIPELINES
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # LOAD DATASET ONCE
    raw_dataset = datasets.ImageFolder(DATA_DIR, transform=None)
    num_samples = len(raw_dataset)

    # K-FOLD SETUP
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(num_samples))):
        print(f"\n🍃 Fold {fold+1}/{NUM_FOLDS} — training begins...")

        train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_transforms), train_idx)
        val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_transforms), val_idx)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = LungCNN(num_classes=3).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images)
                loss = criterion(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(model.state_dict(), f"best_model_fold{fold}.pth")

            print(f"  🌱 Epoch {epoch:02d} — Val Acc: {acc*100:.2f}%")

        elapsed = time.time() - start_time
        print(f"⏳ Fold {fold+1} completed in {elapsed/3600:.2f} hrs — Best Val Acc: {best_val_acc*100:.2f}%")

        # Final evaluation
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall    = recall_score(all_labels, all_preds, average='macro')
        f1        = f1_score(all_labels, all_preds, average='macro')

        fold_metrics.append((accuracy, precision, recall, f1))

    # FINAL OUTPUT
    accs, precs, recs, f1s = zip(*fold_metrics)
    print("\n✨ Final cross-validated results:")
    print(f"  ▶ Accuracy : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  ▶ Precision: {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}%")
    print(f"  ▶ Recall   : {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}%")
    print(f"  ▶ F1-Score : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")


# ——— SAFE WINDOWS MULTIPROCESSING BLOCK ———————————
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    run_training()
