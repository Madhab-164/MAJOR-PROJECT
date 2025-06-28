# 📦 Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.metrics import (DetCurveDisplay, silhouette_samples, silhouette_score, roc_curve,
                              auc, RocCurveDisplay, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set Style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# ----------------------
# 🔵 Data Setup (Mock Example Data) 
# ----------------------
# Normally you load your real data.
# Let's create fake (mock) dataset for demo purpose

# For model predictions
X, y = np.random.randn(500, 20), np.random.randint(0, 2, 500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]  # Probability for ROC/DET
residuals = y_test - y_pred

# ----------------------
# 🔵 1. Bar Plot: Basic Classifiers
# ----------------------
models = ['Naive Bayes', 'AdaBoost', 'Decision Tree', 'KNN', 'LDA']
accuracy = [76.92, 78.61, 90.0, 94.52, 94.92]
f1 = [77.56, 79.45, 92.01, 94.84, 95.97]
precision = [76.92, 83.13, 90.15, 95.83, 95.17]
recall = [76.92, 78.61, 90.09, 94.52, 94.92]

df_basic = pd.DataFrame({'Model': models, 'Accuracy': accuracy, 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

df_basic.set_index('Model').plot(kind='bar', figsize=(10,6))
plt.title('Basic Classifiers Comparison')
plt.ylabel('Percentage (%)')
plt.ylim(60, 100)
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------
# 🔵 2. Line Plot: Vision Transformer Layers
# ----------------------
layers = ['Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9', 'Layer 10']
vit_accuracy = [95.08, 99.27, 96.09, 94.99, 98.09, 99.36]
vit_f1 = [95.5, 99.27, 96.09, 94.99, 98.09, 99.36]
vit_recall = [95.08, 99.27, 96.09, 94.86, 98.05, 99.36]
vit_precision = [94.08, 99.27, 97.29, 94.99, 98.12, 99.36]

plt.figure(figsize=(10,6))
plt.plot(layers, vit_accuracy, marker='o', label='Accuracy')
plt.plot(layers, vit_f1, marker='s', label='F1-Score')
plt.plot(layers, vit_recall, marker='^', label='Recall')
plt.plot(layers, vit_precision, marker='x', label='Precision')
plt.title('Vision Transformer Metrics')
plt.ylabel('Percentage (%)')
plt.ylim(90, 100)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------
# 🔵 3. Radar Chart
# ----------------------
from math import pi

labels = models
metrics_list = [accuracy, f1, precision, recall]  # ✅ Fixed variable name

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
for i, metric in enumerate(metrics_list):
    data = metric + [metric[0]]
    ax.plot(angles, data, label=['Accuracy', 'F1-Score', 'Precision', 'Recall'][i])
    ax.fill(angles, data, alpha=0.1)

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title('Radar Chart - Basic Classifiers')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()

# ----------------------
# 🔵 4. ROC Curve
# ----------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ----------------------
# 🔵 5. DET Curve
# ----------------------
DetCurveDisplay.from_predictions(y_test, y_proba)
plt.title('Detection Error Tradeoff (DET) Curve')
plt.grid(True)
plt.show()

# ----------------------
# 🔵 6. QQ Plot
# ----------------------
plt.figure(figsize=(6, 6))
stats.probplot(residuals, dist="norm", plot=plt)   # ✅ Now works fine
plt.title("QQ Plot of Residuals")
plt.grid(True)
plt.show()

# ----------------------
# 🔵 7. Residual Plot
# ----------------------
sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True)
plt.show()

# ----------------------
# 🔵 8. t-SNE Plot
# ----------------------
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X_test)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y_test, palette='tab10')
plt.title('t-SNE Visualization of Features')
plt.grid(True)
plt.show()

# ----------------------
# 🔵 9. Silhouette Plot
# ----------------------
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_test)

silhouette_avg = silhouette_score(X_test, y_kmeans)
sample_silhouette_values = silhouette_samples(X_test, y_kmeans)

plt.figure(figsize=(8,6))
y_lower = 10
for i in range(2):
    ith_cluster_silhouette_values = sample_silhouette_values[y_kmeans == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values)
    y_lower = y_upper + 10  

plt.title(f'Silhouette Plot (avg = {silhouette_avg:.2f})')
plt.xlabel('Silhouette coefficient')
plt.ylabel('Cluster label')
plt.grid(True)
plt.show()

# ----------------------
# 🔵 10. Learning Curve
# ----------------------
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, label='Training score', marker='o')
plt.plot(train_sizes, test_mean, label='Validation score', marker='o')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------
# 🔵 11. Grid Search Heatmap
# ----------------------
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 7]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)
pivot_table = results.pivot(index='param_max_depth', columns='param_n_estimators', values='mean_test_score')

plt.figure(figsize=(8,6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Grid Search Accuracy Heatmap')
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.show()
