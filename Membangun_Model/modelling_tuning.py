import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Setup MLflow tracking lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Pulm Cancer Prediction Tuning Local")

print("MLflow Tracking URI: file:./mlruns")
print("Untuk melihat hasil: jalankan 'mlflow ui' di terminal")
print("Manual logging digunakan untuk hyperparameter tuning")

data = pd.read_csv("pulmonarycancerclean.csv")

for col in data.columns:
    if col != 'pulmonary_cancer':
        data[col] = data[col].astype('float64')
    else:
        # Keep target variable as int for classification
        data[col] = data[col].astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("pulmonary_cancer", axis=1),
    data["pulmonary_cancer"],
    test_size=0.2,
    random_state=42
)

run_name = f"KNN_Tuning_Modelling"

with mlflow.start_run(run_name=run_name) as run:
    # Manual logging parameter eksperimen
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "accuracy")
    
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

    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Manual logging best parameters
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Manual logging CV score
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)
    
    # Manual logging metrik evaluasi
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
    
    # Manual logging data info
    mlflow.log_metric("data_size", len(data))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    mlflow.log_metric("n_features", X_train.shape[1])
    
    # Manual logging model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_train[:5]
    )
    
    # Manual logging confusion matrix sebagai plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # Manual logging classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    with open('classification_report.json', 'w') as f:
        json.dump(class_report, f, indent=2)
    mlflow.log_artifact('classification_report.json')
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Semua parameter, metrik, dan artifacts telah di-log secara manual ke MLflow")