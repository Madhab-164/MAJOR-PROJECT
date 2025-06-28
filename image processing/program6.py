import pandas as pd
import numpy as np
from PIL import Image
import os
from pycaret.classification import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, validation_curve

dataset_path = r"E:\MAJOR PROJECT\LUNG_PROJECT\New Dataset\Train Case"
classes = ["Normal cases", "Benign cases", "Malignant cases"]

# Store image arrays and labels
data = []
labels = []

for class_name in classes:
    class_dir = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((64, 64))  # Resize to reduce dimensionality
            img_array = np.array(img).flatten()  # Flatten to 1D array
            data.append(img_array)
            labels.append(class_name)
        except Exception as e:
            print(f"Skipped {img_path}: {e}")

# Create DataFrame
df = pd.DataFrame(data)
df['Label'] = labels

# Initialize PyCaret with 3-fold cross-validation
exp = setup(data=df, target='Label', session_id=123, fold=3, verbose=False)

# Compare models including SVM
compared_models = compare_models(include=['svm', 'lr', 'knn', 'rf', 'nb', 'dt'], 
                                 sort='Accuracy', n_select=6)

# Create and tune SVM model
svm = create_model('svm')
tuned_svm = tune_model(svm, optimize='Accuracy')

# ===== Bias-Variance Analysis =====
# 1. Learning Curve
plot_model(tuned_svm, plot='learning', save=True)

# 2. Validation Curve (Manual Implementation)
X = df.drop('Label', axis=1).values
y = df['Label'].values

# Gamma parameter analysis
param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(),
    X,
    y,
    param_name="gamma",
    param_range=param_range,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), 'o-', label="Training Score")
plt.plot(param_range, np.mean(test_scores, axis=1), 'o-', label="CV Score")
plt.xscale('log')
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Validation Curve for SVM')
plt.legend()
plt.savefig('validation_curve.png')
plt.close()

# ===== Final Model Evaluation =====
# Finalize model
final_svm = finalize_model(tuned_svm)

# Model diagnostics
plot_model(final_svm, plot='auc', save=True)
plot_model(final_svm, plot='roc', save=True)
plot_model(final_svm, plot='feature', save=True)

# Generate classification report
predict_model(final_svm)