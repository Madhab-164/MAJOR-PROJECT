import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pycaret.classification import setup, compare_models, pull
from sklearn.ensemble import RandomForestClassifier

# Dataset setup
dataset_path = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Train Case"
categories = ["Normal cases", "Benign cases", "Malignant cases"]
img_size = 128

def load_images_from_folders(base_path, categories):
    data = []
    labels = []
    for idx, category in enumerate(categories):
        folder_path = os.path.join(base_path, category)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                data.append(img.flatten())
                labels.append(category)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(data), np.array(labels)

# Load data
print("🔄 Loading images...")
X, y = load_images_from_folders(dataset_path, categories)
print(f"✅ Loaded {len(X)} images.")

# Create DataFrame
df = pd.DataFrame(X)
df['target'] = y

# Prepare k-fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

# Binarize for ROC/AUC
y_all_bin = label_binarize(y, classes=categories)

for train_index, test_index in kf.split(X, y):
    print(f"\n Fold {fold}")
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_df = X_train.copy()
    train_df['target'] = y_train

    # PyCaret setup
    clf_setup = setup(
    data=train_df,
    target='target',
    session_id=42,
    fold=3,
    log_experiment=False,
    html=False,
    verbose=False
)

    best_model = compare_models(sort='Accuracy')
    results = pull()
    print("\n📊 Evaluation Metrics:")
    print(results[['Model', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1']])

    # Prediction on test
    model = best_model.fit(X_train.drop(columns=['target']), y_train)
    y_pred = model.predict(X_test.drop(columns=['target']))
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test.drop(columns=['target']))
    else:
        y_score = np.zeros((len(y_pred), len(categories)))

    # 🔵 ROC Curve Plot
    plt.figure()
    for i in range(len(categories)):
        fpr, tpr, _ = roc_curve((y_test == categories[i]).astype(int), y_score[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{categories[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"🔵 ROC Curve - Fold {fold}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 🟠 AUC Bar Plot
    auc_scores = []
    for i in range(len(categories)):
        fpr, tpr, _ = roc_curve((y_test == categories[i]).astype(int), y_score[:, i])
        auc_scores.append(auc(fpr, tpr))

    plt.figure()
    plt.bar(categories, auc_scores, color='orange')
    plt.title(f"🟠 AUC Scores - Fold {fold}")
    plt.ylabel("AUC")
    for i, score in enumerate(auc_scores):
        plt.text(i, score + 0.01, f'{score:.2f}', ha='center')
    plt.ylim(0, 1.05)
    plt.grid(axis='y')
    plt.show()

    fold += 1

print("\n✅ All folds completed. Keep going, researcher of life and light! 🌿🧠💡")
