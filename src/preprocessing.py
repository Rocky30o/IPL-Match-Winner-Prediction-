# src/preprocessing.py
"""
Data Preprocessing Script
--------------------------
This script prepares the dataset for machine learning:
1. Formulates the prediction as a binary classification problem:
   - Target = 1 if team1 wins.
   - Target = 0 if team2 wins.
   This prevents the logical error of predicting a team that isn't playing!
2. Splits features (team1, team2, venue, toss_winner, toss_decision, city) and target.
3. Sets up a Preprocessing Pipeline using scikit-learn's ColumnTransformer and OneHotEncoder.
4. Performs an 80:20 Train-Test split.
5. Saves the fitted preprocessor pipeline to 'models/preprocessor_pipeline.pkl'.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, models_dir):
    """Processes features, encodes categorical variables, and splits into train/test sets."""
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n=== Data Preprocessing ===")
    
    # 1. Binary Target Creation
    # y = 1 if team1 wins, y = 0 if team2 wins (winner is always either team1 or team2 in cleaned data)
    # Filter out rows where winner is not team1 or team2 (sanity check)
    df = df[(df['winner'] == df['team1']) | (df['winner'] == df['team2'])].copy()
    
    df['target'] = (df['winner'] == df['team1']).astype(int)
    
    # Print target balance
    target_counts = df['target'].value_counts()
    print(f"[+] Target class distribution (1 = Team 1 Wins, 0 = Team 2 Wins):")
    print(f"    - Team 1 Wins: {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"    - Team 2 Wins: {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")
    
    # Define features and target
    # We select the columns specified in the requirements
    feature_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']
    X = df[feature_cols]
    y = df['target']
    
    # 2. Pipeline Definition
    # We use OneHotEncoder to encode all categorical features.
    # We set handle_unknown='ignore' so that if a new venue or team appears in user input, it won't break.
    categorical_features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the preprocessing pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    # 3. Train-Test Split (80:20)
    # We use stratify=y to ensure target classes are evenly distributed between train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print(f"[+] Train-Test Split Completed (80:20):")
    print(f"    - Training Set: {X_train.shape[0]} samples")
    print(f"    - Testing Set: {X_test.shape[0]} samples")
    
    # Fit the preprocessor on training data
    pipeline.fit(X_train)
    print("[+] Preprocessing pipeline successfully fitted on training data.")
    
    # Save the preprocessing pipeline
    pipeline_path = os.path.join(models_dir, 'preprocessor_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[+] Saved preprocessing pipeline to: {pipeline_path}")
    
    return X_train, X_test, y_train, y_test, pipeline

if __name__ == "__main__":
    # Test run
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import load_data, clean_data
    
    data_path = os.path.join(project_dir, 'data', 'matches.csv')
    models_path = os.path.join(project_dir, 'models')
    try:
        raw_df = load_data(data_path)
        cleaned_df = clean_data(raw_df)
        X_tr, X_te, y_tr, y_te, pipe = preprocess_data(cleaned_df, models_path)
    except Exception as e:
        print(f"[-] Error in Preprocessing: {e}")
