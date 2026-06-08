# src/data_ingestion.py
"""
Data Ingestion & Cleaning Script
--------------------------------
This script handles:
1. Loading the dataset using Pandas.
2. Displaying dataset shape and column info.
3. Standardizing team names (e.g., mapping old team names to their modern names).
4. Handling missing values (e.g., filling missing cities based on venues).
5. Removing irrelevant columns.
"""

import os
import pandas as pd

def load_data(filepath):
    """Loads dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}. Please run download_data.py first.")
    
    print(f"[*] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Cleans and preprocesses the raw IPL matches dataframe."""
    print("\n=== Data Ingestion Info ===")
    print(f"[+] Raw dataset shape: {df.shape}")
    print("\n[+] Raw dataset columns and data types:")
    print(df.dtypes)
    
    # 1. Standardize team names
    # Over the years, team names changed (e.g. Kings XI Punjab to Punjab Kings) or had spelling variants.
    team_mapping = {
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    
    # Apply standardizations
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        df[col] = df[col].replace(team_mapping)
        
    print("\n[*] Standardized team names (mapped old franchises to current names).")
    
    # 2. Handle missing values
    print("\n[+] Missing values count per column before cleaning:")
    print(df.isnull().sum())
    
    # Filter out matches with no winner (No Result / Abandoned matches)
    # Since we want to predict a match winner, we drop matches where 'winner' is null.
    df = df.dropna(subset=['winner'])
    print(f"[+] Dropped matches with no winner. Remaining matches: {len(df)}")
    
    # Impute missing values in 'city'
    # Many missing cities are associated with specific venues. Let's map them.
    venue_city_map = {
        'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
        'Maharashtra Cricket Association Stadium': 'Pune',
        'Saurashtra Cricket Association Stadium': 'Rajkot',
        'Holkar Cricket Stadium': 'Indore',
        'M Chinnaswamy Stadium': 'Bengaluru',
        'Wankhede Stadium': 'Mumbai',
        'Eden Gardens': 'Kolkata',
        'Feroz Shah Kotla': 'Delhi',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali',
        'Green Park': 'Kanpur',
        'Punjab Cricket Association Stadium, Mohali': 'Mohali',
        'Sawai Mansingh Stadium': 'Jaipur',
        'JSCA International Stadium Complex': 'Ranchi',
        'Brabourne Stadium': 'Mumbai',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Visakhapatnam',
        'Subrata Roy Sahara Stadium': 'Pune',
        'Dr DY Patil Sports Academy': 'Mumbai',
        'New Kingsmead Stadium': 'Durban',
        'St George\'s Park': 'Port Elizabeth',
        'Kingsmead': 'Durban',
        'SuperSport Park': 'Centurion',
        'Buffalo Park': 'East London',
        'Newlands': 'Cape Town',
        'De Beers Diamond Oval': 'Kimberley',
        'OUTsurance Oval': 'Bloemfontein',
        'Sharjah Cricket Stadium': 'Sharjah',
        'Dubai International Cricket Stadium': 'Dubai',
        'Sheikh Zayed Stadium': 'Abu Dhabi'
    }
    
    # Fill missing cities based on venue
    missing_city_mask = df['city'].isnull()
    df.loc[missing_city_mask, 'city'] = df.loc[missing_city_mask, 'venue'].map(venue_city_map)
    
    # If still any null cities, fill with 'Unknown'
    df['city'] = df['city'].fillna('Unknown')
    print("[+] Filled missing cities based on venue mapping.")
    
    # 3. Drop irrelevant columns
    # We drop umpire columns, id, date, result type, win_by_runs, win_by_wickets, dl_applied, player_of_match
    # as these are post-match metrics or metadata not available at the start of a match.
    columns_to_drop = ['id', 'date', 'result', 'dl_applied', 'win_by_runs', 
                       'win_by_wickets', 'player_of_match', 'umpire1', 'umpire2', 'umpire3']
    
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"\n[+] Cleaned dataset shape: {df_cleaned.shape}")
    print("[+] Cleaned Columns: ", list(df_cleaned.columns))
    
    return df_cleaned

if __name__ == "__main__":
    # Test execution
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, 'data', 'matches.csv')
    try:
        raw_df = load_data(data_path)
        cleaned_df = clean_data(raw_df)
    except Exception as e:
        print(f"[-] Error: {e}")
