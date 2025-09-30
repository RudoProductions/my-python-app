#!/usr/bin/env python3
"""
Smart Home IDS - Advanced Hybrid GRU + XGBoost Approach
WITH COMPLETE METRICS AND BALANCED DATASET
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, 
    average_precision_score, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import time
import os
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
keras.utils.set_random_seed(RANDOM_STATE)

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = 'IoT_Dataset.csv'
SAVE_DIR = 'saved_models'

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Helper functions
# ----------------------------
def log(title: str):
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all performance metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics

def print_metrics(metrics, model_name):
    """Print formatted metrics"""
    print(f"\n--- {model_name.upper()} PERFORMANCE ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"ROC-AUC: {metrics['auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")

def balance_dataset(X, y, method='smote'):
    """Balance the dataset using various techniques"""
    print(f"Original class distribution: {np.bincount(y)}")
    
    if method == 'smote':
        # SMOTE oversampling
        smote = SMOTE(random_state=RANDOM_STATE)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == 'undersample':
        # Random undersampling
        undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
        X_balanced, y_balanced = undersampler.fit_resample(X, y)
    elif method == 'combined':
        # Combined over and under sampling
        over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE)
        steps = [('o', over), ('u', under)]
        pipeline = ImbPipeline(steps=steps)
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
    else:
        # No balancing
        X_balanced, y_balanced = X, y
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    return X_balanced, y_balanced

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
log("Loading and Preprocessing Data")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded successfully: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find '{DATA_PATH}'. Please update the DATA_PATH variable.")

# Handle non-numeric columns
features_to_drop = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp', 'Cat', 'Sub_Cat']
features_to_drop = [c for c in features_to_drop if c in df.columns]

X = df.drop(columns=features_to_drop + ['Label'])
y = df['Label'].astype(str)

# Convert to numeric and handle missing/inf values
print("Preprocessing features...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Handle infinite values and extreme values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0.0)

# Cap extreme values
for col in X.columns:
    if X[col].dtype in [np.float64, np.int64]:
        finite_vals = X[col][~np.isinf(X[col])]
        if len(finite_vals) > 0:
            q99 = np.percentile(finite_vals, 99.5)
            if q99 > 1e10:
                X[col] = np.clip(X[col], -1e10, q99)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(["Anomaly", "Normal"])
y_encoded = label_encoder.transform(y)
class_names = label_encoder.classes_
print(f"Classes mapped: {dict(zip(class_names, label_encoder.transform(class_names)))}")

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
)

# Balance the training dataset
print("\nBalancing training dataset...")
X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='smote')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Prepare Data for GRU
n_features = X_train_scaled.shape[1]
print(f"\nReshaping data for GRU. Each sample will be: (1, {n_features})")

X_train_gru = X_train_scaled.reshape(X_train_scaled.shape[0], 1, n_features)
X_val_gru = X_val_scaled.reshape(X_val_scaled.shape[0], 1, n_features)
X_test_gru = X_test_scaled.reshape(X_test_scaled.shape[0], 1, n_features)

print(f"GRU Training set shape: {X_train_gru.shape}")
print(f"GRU Validation set shape: {X_val_gru.shape}")
print(f"GRU Test set shape: {X_test_gru.shape}")

# ----------------------------
# 2. Build and Train the GRU Model
# ----------------------------
log("Building and Training GRU Model")

def create_gru_model(input_shape):
    """Creates a GRU model for feature extraction."""
    model = models.Sequential([
        layers.GRU(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.GRU(128, activation='tanh', return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.GRU(256, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    return model

# Create and compile the model
gru_model = create_gru_model((1, n_features))
gru_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

gru_model.summary()

# Calculate class weights for the GRU
gru_class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_balanced),
    y=y_train_balanced
)
gru_class_weights_dict = dict(enumerate(gru_class_weights))

# Train the GRU
print("\nTraining GRU model...")
history = gru_model.fit(
    X_train_gru, y_train_balanced,
    validation_data=(X_val_gru, y_val),
    epochs=30,
    batch_size=256,
    class_weight=gru_class_weights_dict,
    callbacks=[
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
        callbacks.ModelCheckpoint(
            os.path.join(SAVE_DIR, 'gru_model_trained.keras'), 
            save_best_only=True, 
            monitor='val_auc'
        )
    ],
    verbose=1
)

# Save the final model
gru_model.save(os.path.join(SAVE_DIR, 'gru_model_trained.keras'))
print("✓ GRU training completed and model saved.")

# ----------------------------
# 3. Use GRU as Feature Extractor for XGBoost
# ----------------------------
log("Extracting GRU Features for XGBoost")

# Create a feature extraction model: remove the final output layer
feature_extractor = models.Model(
    inputs=gru_model.inputs,
    outputs=gru_model.layers[-3].output # Output from the Dense(128) layer
)

# Extract features from all datasets
print("Extracting features from training set...")
X_train_gru_features = feature_extractor.predict(X_train_gru, verbose=0)
print("Extracting features from validation set...")
X_val_gru_features = feature_extractor.predict(X_val_gru, verbose=0)
print("Extracting features from test set...")
X_test_gru_features = feature_extractor.predict(X_test_gru, verbose=0)

print(f"\nGRU Feature shape: {X_train_gru_features.shape}")

# --- Train XGBoost on GRU Features ---
log("Training XGBoost on GRU Features")

# Handle class imbalance for XGBoost
xgb_class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train_balanced), 
    y=y_train_balanced
)
xgb_scale_pos_weight = xgb_class_weights[1] / xgb_class_weights[0]

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    eval_metric=['logloss', 'auc', 'error'],
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=xgb_scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=0.1,
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_gru_features, y_train_balanced,
    eval_set=[(X_val_gru_features, y_val)],
    verbose=10
)

# Save XGBoost model
xgb_model.save_model(os.path.join(SAVE_DIR, 'xgb_model_gru_features.json'))
print("✓ XGBoost training completed and model saved.")

# ----------------------------
# 4. Build the Hybrid Meta-Model
# ----------------------------
log("Building Hybrid Meta-Model")

# Get predictions from both models on validation set
gru_val_pred_proba = gru_model.predict(X_val_gru, verbose=0).flatten().reshape(-1, 1)
xgb_val_pred_proba = xgb_model.predict_proba(X_val_gru_features)[:, 1].reshape(-1, 1)

# Combine predictions as features for meta-learner
meta_features_val = np.concatenate([gru_val_pred_proba, xgb_val_pred_proba], axis=1)

# Train meta-learner (Logistic Regression)
meta_learner = LogisticRegression(
    max_iter=1000, 
    random_state=RANDOM_STATE,
    C=0.1,
    class_weight='balanced'
)

meta_learner.fit(meta_features_val, y_val)

# Save meta-learner and preprocessing objects
joblib.dump(meta_learner, os.path.join(SAVE_DIR, 'meta_learner.pkl'))
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(SAVE_DIR, 'label_encoder.pkl'))
print("✓ Meta-learner training completed and models saved.")

# ----------------------------
# 5. Evaluate on Test Set
# ----------------------------
log("Final Evaluation on Test Set")

# Get test predictions from both models
gru_test_pred_proba = gru_model.predict(X_test_gru, verbose=0).flatten().reshape(-1, 1)
xgb_test_pred_proba = xgb_model.predict_proba(X_test_gru_features)[:, 1].reshape(-1, 1)

# Combine predictions for meta-learner
meta_features_test = np.concatenate([gru_test_pred_proba, xgb_test_pred_proba], axis=1)

# Hybrid predictions
hybrid_pred_proba = meta_learner.predict_proba(meta_features_test)[:, 1]
hybrid_pred = (hybrid_pred_proba >= 0.5).astype(int)

# Individual model predictions for comparison
gru_test_pred = (gru_test_pred_proba >= 0.5).astype(int).flatten()
xgb_test_pred = (xgb_test_pred_proba >= 0.5).astype(int).flatten()

# ----------------------------
# 6. Results and Analysis
# ----------------------------
log("RESULTS AND ANALYSIS")

# Calculate all metrics for each model
gru_metrics = calculate_all_metrics(y_test, gru_test_pred, gru_test_pred_proba.flatten())
xgb_metrics = calculate_all_metrics(y_test, xgb_test_pred, xgb_test_pred_proba.flatten())
hybrid_metrics = calculate_all_metrics(y_test, hybrid_pred, hybrid_pred_proba)

# Print detailed metrics
print_metrics(gru_metrics, "GRU Model")
print_metrics(xgb_metrics, "XGBoost Model")
print_metrics(hybrid_metrics, "Hybrid Model")

print("\n--- COMPARISON ---")
best_individual_auc = max(gru_metrics['auc'], xgb_metrics['auc'])
best_individual_f1 = max(gru_metrics['f1'], xgb_metrics['f1'])

improvement_auc = ((hybrid_metrics['auc'] - best_individual_auc) / best_individual_auc) * 100
improvement_f1 = ((hybrid_metrics['f1'] - best_individual_f1) / best_individual_f1) * 100

print(f"AUC Improvement: {improvement_auc:+.2f}%")
print(f"F1-Score Improvement: {improvement_f1:+.2f}%")

if improvement_auc > 1.0:
    print("🎉 Hybrid model significantly outperformed individual models!")
elif improvement_auc > 0:
    print("✅ Hybrid model slightly outperformed individual models")
elif improvement_auc >= -1:
    print("🤝 Hybrid model performed comparably to best individual model")
else:
    print("🔍 Individual models performed better")

print("\nDetailed Classification Report (Hybrid Model):")
print(classification_report(y_test, hybrid_pred, target_names=class_names))

# Confusion Matrix
print("\nConfusion Matrix (Hybrid Model):")
cm = confusion_matrix(y_test, hybrid_pred)
print(cm)

# ----------------------------
# 7. Save Configuration with Complete Metrics
# ----------------------------
log("Saving Final Configuration")

# Save configuration with all metrics
config = {
    'feature_order': X.columns.tolist(),
    'label_mapping': {cls: int(label_encoder.transform([cls])[0]) for cls in class_names},
    'model_performance': {
        'gru': gru_metrics,
        'xgb': xgb_metrics,
        'hybrid': hybrid_metrics
    },
    'input_shape': (1, n_features),
    'class_distribution': {
        'original': np.bincount(y_encoded).tolist(),
        'balanced': np.bincount(y_train_balanced).tolist(),
        'test': np.bincount(y_test).tolist()
    },
    'training_params': {
        'balance_method': 'smote',
        'random_state': RANDOM_STATE
    }
}
joblib.dump(config, os.path.join(SAVE_DIR, 'hybrid_gru_config.pkl'))

print("All models and artifacts saved successfully!")
print(f"Files created in '{SAVE_DIR}':")
print("- gru_model_trained.keras")
print("- xgb_model_gru_features.json")
print("- meta_learner.pkl")
print("- scaler.pkl")
print("- label_encoder.pkl")
print("- hybrid_gru_config.pkl")

# ----------------------------
# 8. Create Visualization Plots
# ----------------------------
log("Creating Performance Visualizations")

# Create performance comparison plot
models = ['GRU', 'XGBoost', 'Hybrid']
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    values = [gru_metrics[metric], xgb_metrics[metric], hybrid_metrics[metric]]
    axes[i].bar(models, values, color=['blue', 'green', 'red'])
    axes[i].set_title(f'{metric.upper()} Comparison')
    axes[i].set_ylim(0, 1)
    for j, v in enumerate(values):
        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')

# Hide empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Performance comparison plot saved")

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Hybrid Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("✓ Confusion matrix plot saved")

log("Advanced Hybrid GRU + XGBoost Approach Complete! 🚀")