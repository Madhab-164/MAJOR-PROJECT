import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------
# 1. (Your ViT definitions stay the same)
# -----------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2,0,3,1,4)
        attn = (q @ k.transpose(-2,-1)) / np.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, num_layers, num_heads, mlp_dim, dropout):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        num_patches = (img_size // patch_size) ** 2
        patch_dim   = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token   = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.patch_size = patch_size
        self.img_size   = img_size

    def forward(self, x):
        B, C, H, W = x.shape
        patches = (
            x.unfold(2,self.patch_size,self.patch_size)
             .unfold(3,self.patch_size,self.patch_size)
             .permute(0,2,3,1,4,5)
             .reshape(B, -1, C*self.patch_size**2)
        )
        x = self.patch_embed(patches)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:,0])

# -----------------------------------
# 2. Training / Evaluation (unchanged)
# -----------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, samples = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct   += (preds==labels).sum().item()
        samples   += imgs.size(0)
    return total_loss/samples, correct/samples

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds  = logits.argmax(dim=1).cpu().numpy()
            ps.extend(preds); ys.extend(labels.numpy())
    acc  = accuracy_score(ys, ps)
    prec = precision_score(ys, ps, average='weighted', zero_division=0)
    rec  = recall_score(ys, ps, average='weighted', zero_division=0)
    f1   = f1_score(ys, ps, average='weighted', zero_division=0)
    return acc, prec, rec, f1

# -----------------------------------
# 3. K‑Fold Cross‑Validation & Main
# -----------------------------------
def run_kfold(data_dir,
              img_size=224, 
              patch_size=16, 
              in_channels=3,
              num_layers=4, 
              embed_dim=256, 
              num_heads=8, 
              mlp_dim=512,
              dropout=0.1, 
              batch_size=32, 
              num_epochs=10, 
              lr=1e-4,
              k=5, 
              device=None):
    # Device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augmented transforms
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Full dataset
    full_ds = datasets.ImageFolder(data_dir, transform=transform)
    class_names = full_ds.classes
    indices = np.arange(len(full_ds))

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n🌿 Fold {fold}/{k} — training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        # Subsets & loaders
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=batch_size, shuffle=False, num_workers=4)

        # Model / Loss / Opt / Scheduler
        model = VisionTransformer(img_size, patch_size, in_channels, len(class_names),
                                  embed_dim, num_layers, num_heads, mlp_dim, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Per‑fold timing
        start_fold = time.time()
        for epoch in range(1, num_epochs+1):
            t0 = time.time()
            loss, acc_tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
            acc_val, prec, rec, f1 = evaluate(model, val_loader, device)
            scheduler.step()
            print(f" Epoch {epoch:02d}/{num_epochs} — "
                  f"train_loss: {loss:.4f}, train_acc: {acc_tr*100:.2f}% | "
                  f"val_acc: {acc_val*100:.2f}%, val_f1: {f1*100:.2f}% | "
                  f"time: {(time.time()-t0)/60:.2f} min")
        print(f"Fold {fold} done in {(time.time()-start_fold)/60:.2f} min")
        fold_metrics.append((acc_val, prec, rec, f1))

    # Aggregate
    accs, precs, recs, f1s = zip(*fold_metrics)
    print("\n✨ Cross‑Validation Results ✨")
    print(f" Mean Accuracy : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f" Mean Precision: {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}%")
    print(f" Mean Recall   : {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}%")
    print(f" Mean F1 Score : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")

    # Final confusion on last fold
    # (you could also re-evaluate on a held‑out test set here)
    return fold_metrics, class_names

if __name__ == '__main__':
    DATA_DIR   = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Train Case"
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run K‑Fold CV
    metrics, classes = run_kfold(DATA_DIR, device=DEVICE)
