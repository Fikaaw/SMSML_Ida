import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub
import os

# Inisialisasi DagsHub dan MLflow tracking
dagshub.init(repo_owner='Fikaaw', repo_name='pulm_cancer_modelling_experiment_tracking', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")

print(f"MLflow Tracking URI: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking.mlflow")
print("Untuk melihat hasil: https://dagshub.com/Fikaaw/pulm_cancer_modelling_experiment_tracking")

mlflow.set_experiment("Pulm Cancer Prediction Tuning")

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
    test_size=0.2,
    random_state=42
)

run_name = f"KNN_Tuning_Modelling"

with mlflow.start_run(run_name=run_name) as run:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("All parameters, metrics, and model artifacts logged automatically via autolog")