import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score, 
    matthews_corrcoef, balanced_accuracy_score, log_loss
)
import dagshub
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc

# Setup DagsHub dan MLflow tracking online
dagshub.init(repo_owner='Fikaaw', repo_name='pulm_cancer_modelling_experiment_tracking', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")

print(f"MLflow Tracking URI: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")
print("Untuk melihat hasil: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking")
print("Manual logging dengan metrik tambahan untuk DagsHub")

mlflow.set_experiment("Pulm Cancer Prediction Advanced DagsHub")

# Load dan preprocessing data
data = pd.read_csv("pulmonarycancerclean.csv")

for col in data.columns:
    if col != 'pulmonary_cancer':
        data[col] = data[col].astype('float64')
    else:
        data[col] = data[col].astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("pulmonary_cancer", axis=1),
    data["pulmonary_cancer"],
    test_size=0.2,
    random_state=42,
    stratify=data["pulmonary_cancer"]
)

run_name = f"KNN_Advanced_DagsHub_Manual_Logging"

with mlflow.start_run(run_name=run_name) as run:
    start_time = time.time()
    
    # Manual logging parameter eksperimen
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "accuracy")
    mlflow.log_param("stratify", True)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Log parameter grid
    mlflow.log_param("param_grid", str(param_grid))
    mlflow.log_param("total_param_combinations", len(param_grid['knn__n_neighbors']) * len(param_grid['knn__weights']) * len(param_grid['knn__metric']))

    grid_search = GridSearchCV(
        pipeline, param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # Manual logging best parameters
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Manual logging CV scores
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("cv_std", grid_search.cv_results_['std_test_score'][grid_search.best_index_])
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # ===== METRIK STANDAR AUTOLOG =====
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    
    # ===== METRIK TAMBAHAN 1: ROC AUC SCORE =====
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    
    # ===== METRIK TAMBAHAN 2: MATTHEWS CORRELATION COEFFICIENT =====
    test_mcc = matthews_corrcoef(y_test, y_pred)
    mlflow.log_metric("test_matthews_corrcoef", test_mcc)
    
    # ===== METRIK TAMBAHAN 3: BALANCED ACCURACY =====
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_balanced_accuracy", test_balanced_acc)
    
    # ===== METRIK TAMBAHAN 4: LOG LOSS =====
    test_log_loss = log_loss(y_test, y_pred_proba)
    mlflow.log_metric("test_log_loss", test_log_loss)
    
    # ===== METRIK TAMBAHAN 5: OVERFITTING INDICATOR =====
    overfitting_score = train_accuracy - test_accuracy
    mlflow.log_metric("overfitting_score", overfitting_score)
    
    # Manual logging data info
    mlflow.log_metric("data_size", len(data))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    mlflow.log_metric("n_features", X_train.shape[1])
    mlflow.log_metric("class_balance_ratio", y_train.value_counts().min() / y_train.value_counts().max())
    
    # Training time
    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    
    # Manual logging model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_train[:5]
    )
    
    # ===== ARTIFACTS TAMBAHAN =====
    
    # 1. Confusion Matrix dengan lebih detail
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'])
    plt.title(f'Confusion Matrix\nAccuracy: {test_accuracy:.4f}, ROC-AUC: {test_roc_auc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_advanced.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('confusion_matrix_advanced.png')
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('roc_curve.png')
    plt.close()
    
    # 3. Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='blue')
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('learning_curve.png')
    plt.close()
    
    # 4. Feature Importance (untuk KNN, kita gunakan permutation importance)
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance_mean'], 
             xerr=importance_df['importance_std'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance (Permutation)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # 5. Classification Report dengan metrik tambahan
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Tambahkan metrik custom ke report
    class_report['custom_metrics'] = {
        'roc_auc_score': test_roc_auc,
        'matthews_corrcoef': test_mcc,
        'balanced_accuracy': test_balanced_acc,
        'log_loss': test_log_loss,
        'overfitting_score': overfitting_score,
        'training_time_seconds': training_time
    }
    
    with open('classification_report_advanced.json', 'w') as f:
        json.dump(class_report, f, indent=2)
    mlflow.log_artifact('classification_report_advanced.json')
    
    # 6. Model Performance Summary
    performance_summary = {
        'model_type': 'KNeighborsClassifier',
        'best_parameters': grid_search.best_params_,
        'standard_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1
        },
        'advanced_metrics': {
            'roc_auc': test_roc_auc,
            'matthews_corrcoef': test_mcc,
            'balanced_accuracy': test_balanced_acc,
            'log_loss': test_log_loss,
            'overfitting_score': overfitting_score
        },
        'training_info': {
            'training_time_seconds': training_time,
            'cv_score': grid_search.best_score_,
            'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'total_combinations_tested': len(param_grid['knn__n_neighbors']) * len(param_grid['knn__weights']) * len(param_grid['knn__metric'])
        }
    }
    
    with open('model_performance_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    mlflow.log_artifact('model_performance_summary.json')
    
    print("=== HASIL TRAINING MODEL ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f} ± {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.4f}")
    print("\n=== METRIK STANDAR ===")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("\n=== METRIK TAMBAHAN ===")
    print(f"ROC AUC Score: {test_roc_auc:.4f}")
    print(f"Matthews Correlation Coefficient: {test_mcc:.4f}")
    print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f"Log Loss: {test_log_loss:.4f}")
    print(f"Overfitting Score: {overfitting_score:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print("\nSemua parameter, metrik standar + tambahan, dan artifacts telah di-log ke DagsHub MLflow")