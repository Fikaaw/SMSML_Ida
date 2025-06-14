import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import dagshub
import mlflow.sklearn
import os

# Inisialisasi DagsHub dan MLflow tracking
dagshub.init(repo_owner='Fikaaw', repo_name='pulm_cancer_modelling_experiment_tracking', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")

print(f"MLflow Tracking URI: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")
print("Untuk melihat hasil: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking")

mlflow.set_experiment("Pulm Cancer Prediction")

# Aktifkan autolog untuk automatic logging ke DagsHub
mlflow.sklearn.autolog()

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
    # Contoh logging parameter dan metrik
    n_neighbors = 5
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", "auto")
    mlflow.log_param("weights", "uniform")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    model.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Calculate and log metrics
    accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("data_size", len(data))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
