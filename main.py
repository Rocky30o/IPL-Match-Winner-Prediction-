# main.py
"""
End-to-End IPL Match Winner Prediction Pipeline
----------------------------------------------
This is the master script that runs the entire machine learning pipeline:
1. Downloads the IPL matches dataset.
2. Ingests and cleans the data (standardizing names, handling nulls).
3. Performs Exploratory Data Analysis (EDA) and saves plots.
4. Preprocesses features and sets up the preprocessing pipeline.
5. Trains Machine Learning models (Logistic Regression & Random Forest).
6. Evaluates models and saves confusion matrices & feature importance charts.
7. Executes a sample match prediction (Mumbai Indians vs Chennai Super Kings).
8. Programmatically compiles the student-ready Jupyter Notebook.
"""

import os
import sys

# Append the src directory to Python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import pipeline steps
from download_data import download_dataset
from data_ingestion import load_data, clean_data
from eda import run_eda
from preprocessing import preprocess_data
from model_training import train_models
from evaluation import evaluate_models
from predict import predict_match_winner, print_prediction_report
from generate_notebook import generate_notebook

def run_pipeline():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Define paths
    csv_path = os.path.join(project_dir, 'data', 'matches.csv')
    models_dir = os.path.join(project_dir, 'models')
    outputs_dir = os.path.join(project_dir, 'outputs')
    
    print("="*60)
    print("      STARTING IPL MATCH WINNER PREDICTION PIPELINE      ")
    print("="*60)
    
    # Step 1: Download dataset
    print("\n--- STEP 1: DOWNLOADING DATASET ---")
    download_success = download_dataset()
    if not download_success:
        print("[-] Pipeline failed during data download.")
        return
        
    # Step 2: Load and clean dataset
    print("\n--- STEP 2: DATA INGESTION & CLEANING ---")
    raw_df = load_data(csv_path)
    cleaned_df = clean_data(raw_df)
    
    # Step 3: Run EDA & Save Visualizations
    print("\n--- STEP 3: EXPLORATORY DATA ANALYSIS (EDA) ---")
    run_eda(cleaned_df, outputs_dir)
    
    # Step 4: Preprocess Data
    print("\n--- STEP 4: PREPROCESSING & PIPELINE SETUP ---")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(cleaned_df, models_dir)
    
    # Step 5: Train Models
    print("\n--- STEP 5: MODEL TRAINING (LOGISTIC REGRESSION & RANDOM FOREST) ---")
    lr_model, rf_model = train_models(X_train, y_train, preprocessor, models_dir)
    
    # Step 6: Evaluate Models
    print("\n--- STEP 6: MODEL EVALUATION & FEATURE IMPORTANCE ---")
    evaluate_models(X_test, y_test, lr_model, rf_model, preprocessor, outputs_dir)
    
    # Step 7: Sample Prediction
    print("\n--- STEP 7: RUNNING SAMPLE PREDICTION SCENARIO ---")
    # Predicting Mumbai Indians vs Chennai Super Kings at Wankhede Stadium with MI winning toss and fielding
    report = predict_match_winner(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="field",
        models_dir=models_dir
    )
    print_prediction_report(report)
    
    # Step 8: Generate Jupyter Notebook
    print("\n--- STEP 8: GENERATING EDUCATION-READY JUPYTER NOTEBOOK ---")
    generate_notebook()
    
    print("\n" + "="*60)
    print("     IPL MATCH WINNER PREDICTION PIPELINE COMPLETED!     ")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
