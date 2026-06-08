# src/predict.py
"""
Match Winner Prediction System
------------------------------
This script defines a prediction function that allows a user to input:
- Team 1
- Team 2
- Venue
- Toss Winner
- Toss Decision

And receives:
1. Predicted Winner
2. Winning Probability for both teams

It automatically resolves the city based on the input venue to simplify user inputs.
"""

import os
import pickle
import pandas as pd

# Reuse the venue to city mapping from data ingestion to simplify user input
VENUE_CITY_MAP = {
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

def load_prediction_assets(models_dir):
    """Loads the preprocessor pipeline and trained models."""
    preprocessor_path = os.path.join(models_dir, 'preprocessor_pipeline.pkl')
    model_path = os.path.join(models_dir, 'random_forest_model.pkl') # Use Random Forest as default
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Models or preprocessor not found. Please run model_training.py first.")
        
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return preprocessor, model

def predict_match_winner(team1, team2, venue, toss_winner, toss_decision, models_dir):
    """Predicts the match winner and probability for a given matchup."""
    preprocessor, model = load_prediction_assets(models_dir)
    
    # 1. Validation & Input Cleaning
    team1 = team1.strip()
    team2 = team2.strip()
    venue = venue.strip()
    toss_winner = toss_winner.strip()
    toss_decision = toss_decision.strip().lower()
    
    if toss_decision not in ['bat', 'field']:
        raise ValueError("Toss decision must be either 'bat' or 'field'.")
        
    if toss_winner not in [team1, team2]:
        raise ValueError("Toss winner must be either Team 1 or Team 2.")
        
    # Resolve City based on Venue
    city = VENUE_CITY_MAP.get(venue, 'Unknown')
    
    # 2. Prepare Input Dataframe
    # Feature columns must match what the preprocessing pipeline expects
    input_data = pd.DataFrame([{
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'city': city
    }])
    
    # 3. Transform input using pipeline
    input_transformed = preprocessor.transform(input_data)
    
    # 4. Predict
    # Get probability output
    # proba is a 2D array: [[prob_class_0, prob_class_1]]
    # class 1 = Team 1 Wins, class 0 = Team 2 Wins
    probabilities = model.predict_proba(input_transformed)[0]
    prob_team2_win = probabilities[0]
    prob_team1_win = probabilities[1]
    
    # Determine winner
    if prob_team1_win >= prob_team2_win:
        predicted_winner = team1
        winning_prob = prob_team1_win
    else:
        predicted_winner = team2
        winning_prob = prob_team2_win
        
    results = {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'city': city,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'predicted_winner': predicted_winner,
        'winning_probability': winning_prob,
        'team1_probability': prob_team1_win,
        'team2_probability': prob_team2_win
    }
    
    return results

def print_prediction_report(res):
    """Utility to print a beautiful output report for the student's demonstration."""
    print("\n" + "="*50)
    print("           IPL MATCH WINNER PREDICTION          ")
    print("="*50)
    print(f"Matchup:       {res['team1']} vs {res['team2']}")
    print(f"Venue:         {res['venue']} (City: {res['city']})")
    print(f"Toss Winner:   {res['toss_winner']} ({res['toss_decision']} first)")
    print("-"*50)
    print(f"Predicted Winner:    \033[1;32m{res['predicted_winner']}\033[0m")
    print(f"Winning Probability: \033[1;36m{res['winning_probability']*100:.2f}%\033[0m")
    print("-"*50)
    print(f"Win Probability breakdown:")
    print(f"   - {res['team1']}: {res['team1_probability']*100:.2f}%")
    print(f"   - {res['team2']}: {res['team2_probability']*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test Prediction Run
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(project_dir, 'models')
    
    # Sample Test Scenario: Mumbai Indians vs Chennai Super Kings
    try:
        report = predict_match_winner(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            toss_winner="Mumbai Indians",
            toss_decision="field",
            models_dir=models_path
        )
        print_prediction_report(report)
    except Exception as e:
        print(f"[-] Prediction failed (Ensure you trained models first): {e}")
