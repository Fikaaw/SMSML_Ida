import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import mlflow.sklearn

# Setup MLflow tracking lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Pulm Cancer Prediction Local")

# Aktifkan autolog untuk automatic logging secara lokal
mlflow.sklearn.autolog()

print("MLflow Tracking URI: file:./mlruns")
print("Untuk melihat hasil: jalankan 'mlflow ui' di terminal")

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
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

run_name = f"KNN_Base_Modelling"

with mlflow.start_run(run_name=run_name):
    # Training model dengan autolog aktif - semua parameter, metrik, dan model akan di-log otomatis
    model = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
    model.fit(X_train, y_train)
    
    # Evaluasi model - metrik akan di-log otomatis oleh autolog
    accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print("Semua parameter, metrik, dan model telah di-log otomatis oleh MLflow autolog")
