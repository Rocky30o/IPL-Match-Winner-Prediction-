# src/model_training.py
"""
Model Training Script
---------------------
This script trains two classification models:
1. Logistic Regression: A simple, linear model that serves as our baseline.
2. Random Forest Classifier: An ensemble tree-based model that can capture non-linear relationships.

Trained models are saved as pickle files under the 'models/' directory.
"""

import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train, preprocessor_pipeline, models_dir):
    """Transforms features using the preprocessor pipeline and trains ML models."""
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n=== Model Training ===")
    
    # 1. Transform features using the saved preprocessor pipeline
    # We transform the raw categorical dataframe into a numerical one-hot encoded matrix.
    print("[*] Transforming training features...")
    X_train_transformed = preprocessor_pipeline.transform(X_train)
    print(f"[+] Transformed training shape: {X_train_transformed.shape}")
    
    # 2. Train Logistic Regression
    # We increase max_iter to ensure convergence of the gradient descent solver.
    print("[*] Training Logistic Regression Classifier...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_transformed, y_train)
    print("[+] Logistic Regression model trained.")
    
    # Save Logistic Regression model
    lr_model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"[+] Saved Logistic Regression model to: {lr_model_path}")
    
    # 3. Train Random Forest Classifier
    # We use n_estimators=100 (100 decision trees) and set random_state for reproducibility.
    print("[*] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_transformed, y_train)
    print("[+] Random Forest model trained.")
    
    # Save Random Forest model
    rf_model_path = os.path.join(models_dir, 'random_forest_model.pkl')
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[+] Saved Random Forest model to: {rf_model_path}")
    
    return lr_model, rf_model

if __name__ == "__main__":
    # Test execution
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import load_data, clean_data
    from preprocessing import preprocess_data
    
    data_path = os.path.join(project_dir, 'data', 'matches.csv')
    models_path = os.path.join(project_dir, 'models')
    try:
        raw_df = load_data(data_path)
        cleaned_df = clean_data(raw_df)
        X_tr, X_te, y_tr, y_te, pipe = preprocess_data(cleaned_df, models_path)
        train_models(X_tr, y_tr, pipe, models_path)
    except Exception as e:
        print(f"[-] Error in training: {e}")
