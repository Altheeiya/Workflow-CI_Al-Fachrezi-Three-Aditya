import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import argparse
import os

mlflow.set_tracking_uri("http://localhost:5000")


def load_data(data_dir: str) -> tuple:
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test


def train_and_log(n_estimators: int, max_depth: int):
    data_dir = 'telco_churn_preprocessing'
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        print(f"Training with n_estimators={n_estimators}, max_depth={max_depth}")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        # Explicit log model
        mlflow.sklearn.log_model(model, "model")
        run = mlflow.active_run()
        model_path = os.path.join(mlflow.get_artifact_uri(), "model")
        print(f"Model logged successfully at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train_and_log(args.n_estimators, args.max_depth)
