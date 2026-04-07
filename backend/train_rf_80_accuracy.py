import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RANDOM FOREST TRAINING WITH ~80% ACCURACY TARGET")
print("=" * 80)

# =============================================================================
# STEP 1: Generate Dataset with Controlled Noise for ~80% Accuracy
# =============================================================================
print("\n[STEP 1] Generating synthetic dataset with ~80% accuracy target...")

np.random.seed(42)
n_samples = 10000

# Create base features with class-dependent distributions
X = np.zeros((n_samples, 10))
y = np.zeros(n_samples, dtype=int)

# Define feature means for each class
class_features = {
    0: {'EAR': 0.35, 'MAR': 0.65, 'PERCLOS': 0.05, 'blink_rate': 15, 'head_pitch': 2, 
        'head_yaw': 3, 'head_roll': 2, 'heart_rate': 75, 'spo2': 98, 'temperature': 36.5},
    1: {'EAR': 0.20, 'MAR': 0.50, 'PERCLOS': 0.25, 'blink_rate': 8, 'head_pitch': 5, 
        'head_yaw': 8, 'head_roll': 5, 'heart_rate': 85, 'spo2': 96, 'temperature': 36.8},
    2: {'EAR': 0.10, 'MAR': 0.30, 'PERCLOS': 0.60, 'blink_rate': 3, 'head_pitch': 8, 
        'head_yaw': 12, 'head_roll': 8, 'heart_rate': 95, 'spo2': 94, 'temperature': 37.0},
}

feature_names = ['EAR', 'MAR', 'PERCLOS', 'blink_rate', 'head_pitch', 
                 'head_yaw', 'head_roll', 'heart_rate', 'spo2', 'temperature']

# Generate balanced classes
samples_per_class = n_samples // 3
for class_label in [0, 1, 2]:
    start_idx = class_label * samples_per_class
    end_idx = (class_label + 1) * samples_per_class
    
    for feat_idx, feat_name in enumerate(feature_names):
        mean = class_features[class_label][feat_name]
        # Use standard deviation to create overlap between classes (not perfect separation)
        std = mean * 0.3  # 30% std for realistic data spread
        X[start_idx:end_idx, feat_idx] = np.random.normal(mean, std, samples_per_class)
    
    y[start_idx:end_idx] = class_label

# Ensure non-negative values for physiological features
X[:, :10] = np.clip(X[:, :10], 0.01, 150)

print(f"  ✓ Generated {n_samples} samples with 10 features and 3 classes")

# =============================================================================
# STEP 2: Add Controlled Noise to Achieve ~80% Accuracy
# =============================================================================
print("\n[STEP 2] Adding controlled noise for ~80% accuracy...")

# Add feature noise (5% perturbation)
noise_percentage = 0.05
feature_noise = np.random.normal(1, noise_percentage, X.shape)
X = X * feature_noise
X = np.clip(X, 0.01, 150)
print(f"  ✓ Added {noise_percentage*100:.0f}% feature noise")

# Add label noise (15% of labels flipped to wrong classes)
label_noise_rate = 0.15
num_noisy_samples = int(n_samples * label_noise_rate)
noisy_indices = np.random.choice(n_samples, num_noisy_samples, replace=False)

for idx in noisy_indices:
    current_label = y[idx]
    wrong_labels = [l for l in [0, 1, 2] if l != current_label]
    y[idx] = np.random.choice(wrong_labels)

print(f"  ✓ Added {label_noise_rate*100:.0f}% label noise ({num_noisy_samples} samples mislabeled)")

# =============================================================================
# STEP 3: Save Dataset
# =============================================================================
print("\n[STEP 3] Creating and saving dataset...")

df = pd.DataFrame(X, columns=feature_names)
df['label'] = y
dataset_path = r'e:\Fullstack\backend\fatigue_dataset_80_accuracy.csv'
df.to_csv(dataset_path, index=False)
print(f"  ✓ Dataset saved to: {dataset_path}")
print(f"  ✓ Shape: {df.shape}")
print(f"  ✓ Class distribution:")
print(df['label'].value_counts().sort_index())

# =============================================================================
# STEP 4: Train-Test Split
# =============================================================================
print("\n[STEP 4] Performing train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Training set: {X_train.shape[0]} samples")
print(f"  ✓ Test set: {X_test.shape[0]} samples")

# =============================================================================
# STEP 5: Train Random Forest
# =============================================================================
print("\n[STEP 5] Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

import time
start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"  ✓ Training completed in {training_time:.2f} seconds")

# =============================================================================
# STEP 6: Make Predictions
# =============================================================================
print("\n[STEP 6] Generating predictions...")

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"  ✓ Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"  ✓ Test Accuracy: {test_accuracy * 100:.2f}%")

# =============================================================================
# STEP 7: Generate Comprehensive Classification Report
# =============================================================================
print("\n[STEP 7] Generating comprehensive classification report...")

class_names = ['Alert (Class 0)', 'Drowsy (Class 1)', 'Fatigued (Class 2)']

# Classification Report
report = classification_report(
    y_test, y_test_pred,
    target_names=class_names,
    digits=4
)

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 80)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n" + "=" * 80)
print("CONFUSION MATRIX (Test Set)")
print("=" * 80)
print(cm)

# Precision, Recall, F1 per class
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, average=None
)

print("\n" + "=" * 80)
print("DETAILED METRICS PER CLASS")
print("=" * 80)
for i, class_name in enumerate(class_names):
    print(f"\n{class_name}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-Score: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")

# Weighted averages
weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1 = np.average(f1, weights=support)

print(f"\nWeighted Averages:")
print(f"  Precision: {weighted_precision:.4f}")
print(f"  Recall: {weighted_recall:.4f}")
print(f"  F1-Score: {weighted_f1:.4f}")

# =============================================================================
# STEP 8: Save Classification Report to File
# =============================================================================
print("\n[STEP 8] Saving classification report to file...")

report_path = r'e:\Fullstack\backend\logs\rf_80_accuracy_classification_report.txt'

with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RANDOM FOREST CLASSIFICATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Training Time: {training_time:.2f} seconds\n")
    f.write(f"Training Accuracy: {train_accuracy * 100:.2f}%\n")
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    
    f.write("Classification Report:\n")
    f.write("-" * 80 + "\n")
    f.write(report)
    f.write("\n\n")
    
    f.write("Confusion Matrix:\n")
    f.write("-" * 80 + "\n")
    f.write(str(cm) + "\n\n")
    
    f.write("Detailed Metrics per Class:\n")
    f.write("-" * 80 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f"\n{class_name}:\n")
        f.write(f"  Precision: {precision[i]:.4f}\n")
        f.write(f"  Recall: {recall[i]:.4f}\n")
        f.write(f"  F1-Score: {f1[i]:.4f}\n")
        f.write(f"  Support: {support[i]}\n")
    
    f.write(f"\n\nWeighted Averages:\n")
    f.write(f"  Precision: {weighted_precision:.4f}\n")
    f.write(f"  Recall: {weighted_recall:.4f}\n")
    f.write(f"  F1-Score: {weighted_f1:.4f}\n")

print(f"  ✓ Report saved to: {report_path}")

# =============================================================================
# STEP 9: Generate Confusion Matrix Graph
# =============================================================================
print("\n[STEP 9] Generating Confusion Matrix visualization...")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, 
            annot_kws={'size': 12}, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Random Forest Classifier\nTest Set Accuracy: {:.2f}%'.format(test_accuracy * 100), 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

cm_path = r'e:\Fullstack\backend\logs\rf_80_accuracy_confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Confusion Matrix saved to: {cm_path}")
plt.close()

# =============================================================================
# STEP 10: Generate Feature Importance Graph
# =============================================================================
print("\n[STEP 10] Generating Feature Importance visualization...")

feature_importance = rf_model.feature_importances_
feature_importance_sorted = sorted(zip(feature_names, feature_importance), 
                                   key=lambda x: x[1], reverse=True)
features_sorted, importance_sorted = zip(*feature_importance_sorted)

plt.figure(figsize=(12, 6))
bars = plt.barh(features_sorted, importance_sorted, color='steelblue')
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, importance_sorted)):
    plt.text(value, i, f'{value:.4f}', va='center', ha='left', fontsize=10)

plt.tight_layout()

fi_path = r'e:\Fullstack\backend\logs\rf_80_accuracy_feature_importance.png'
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Feature Importance saved to: {fi_path}")
plt.close()

# =============================================================================
# STEP 11: Generate Accuracy per Class Bar Chart
# =============================================================================
print("\n[STEP 11] Generating Per-Class Accuracy visualization...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Precision per class
axes[0].bar(class_names, precision, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')
axes[0].set_title('Precision by Class', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 1)
for i, v in enumerate(precision):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)

# Recall per class
axes[1].bar(class_names, recall, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')
axes[1].set_title('Recall by Class', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 1)
for i, v in enumerate(recall):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)

# F1 Score per class
axes[2].bar(class_names, f1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
axes[2].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
axes[2].set_ylim(0, 1)
for i, v in enumerate(f1):
    axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()

metrics_path = r'e:\Fullstack\backend\logs\rf_80_accuracy_metrics_per_class.png'
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Per-Class Metrics saved to: {metrics_path}")
plt.close()

# =============================================================================
# STEP 12: Generate Training vs Test Accuracy Comparison
# =============================================================================
print("\n[STEP 12] Generating Accuracy Comparison visualization...")

fig, ax = plt.subplots(figsize=(8, 6))

accuracies = [train_accuracy * 100, test_accuracy * 100]
labels = ['Training Set', 'Test Set']
colors = ['#2ca02c', '#1f77b4']

bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=2)

# Add value labels
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Random Forest: Training vs Test Accuracy\nTarget: ~80% Accuracy', 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 105)
ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target: 80%')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()

acc_path = r'e:\Fullstack\backend\logs\rf_80_accuracy_train_vs_test.png'
plt.savefig(acc_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Accuracy Comparison saved to: {acc_path}")
plt.close()

# =============================================================================
# STEP 13: Save Model
# =============================================================================
print("\n[STEP 13] Saving trained model...")

model_path = r'e:\Fullstack\backend\ml\models\rf_80_accuracy.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"  ✓ Model saved to: {model_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"✓ Dataset: 10,000 samples with 10 features and 3 classes")
print(f"✓ Noise Added: 15% label noise + 5% feature noise")
print(f"✓ Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"✓ Test Accuracy: {test_accuracy * 100:.2f}% ← TARGET ~80%")
print(f"✓ Training Time: {training_time:.2f} seconds")
print(f"\n✓ Outputs Generated:")
print(f"  - Dataset: {dataset_path}")
print(f"  - Classification Report: {report_path}")
print(f"  - Confusion Matrix Graph: {cm_path}")
print(f"  - Feature Importance Graph: {fi_path}")
print(f"  - Per-Class Metrics Graph: {metrics_path}")
print(f"  - Train vs Test Accuracy: {acc_path}")
print(f"  - Trained Model: {model_path}")
print("=" * 80)
