# app.py
"""
IPL Match Winner Prediction Web Application
-------------------------------------------
A premium, interactive Streamlit frontend for the IPL Match Winner Prediction pipeline.
Features:
1. Tab 1: Match Predictor - Select teams, venue, toss details, and model to get real-time winning probabilities.
2. Tab 2: Historical Insights - View interactive visualizations generated during EDA.
3. Tab 3: Model Diagnostics - Compare Logistic Regression & Random Forest classifiers.
"""

import os
import sys
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Setup Python path to import modules from src
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_dir, 'src'))

from data_ingestion import load_data, clean_data
from predict import predict_match_winner

# Set Page Config
st.set_page_config(
    page_title="IPL Match Winner Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling / Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Global Overrides */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        color: white;
        margin-bottom: 25px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .header-card h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 5px;
        letter-spacing: 1px;
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-card p {
        font-size: 1.15rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Prediction Glassmorphism Card */
    .prediction-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Winner Display card */
    .winner-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        color: white;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    .winner-card h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        margin: 5px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Custom progress bar wrapper */
    .prob-bar-label {
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 1rem;
    }
    .prob-bar-outer {
        background-color: #333333;
        border-radius: 10px;
        width: 100%;
        height: 24px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .prob-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    .prob-label-value {
        position: absolute;
        right: 12px;
        top: 2px;
        font-weight: bold;
        color: white;
        font-size: 13px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load data to get unique values dynamically (cached for efficiency)
@st.cache_data
def get_clean_dataset():
    csv_path = os.path.join(project_dir, 'data', 'matches.csv')
    raw_df = load_data(csv_path)
    return clean_data(raw_df)

try:
    df_cleaned = get_clean_dataset()
    unique_teams = sorted(df_cleaned['team1'].unique().tolist())
    unique_venues = sorted(df_cleaned['venue'].unique().tolist())
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Make sure the pipeline dataset downloading and preprocessing is completed first.")
    st.stop()

# Header Display
st.markdown("""
<div class="header-card">
    <h1>🏏 IPL MATCH WINNER PREDICTOR</h1>
    <p>Predict match winner probabilities based on team history, toss decisions, and venues using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Create Tabs
tab_predict, tab_eda, tab_metrics = st.tabs([
    "🔮 Match Predictor",
    "📊 Historical Insights (EDA)",
    "⚙️ Model Diagnostics & Metrics"
])

# Sidebar Controls for general app configuration
with st.sidebar:
    st.markdown("### 🛠️ Model Settings")
    model_choice = st.selectbox(
        "Choose Classification Model",
        ["Random Forest Classifier", "Logistic Regression"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.write("1. Select the playing teams in the Predictor tab.")
    st.write("2. Select the toss outcomes and venue.")
    st.write("3. Click **Predict Winner** to view probabilities.")
    st.write("4. Navigate to other tabs to inspect data patterns and model metrics.")

# -----------------
# TAB 1: PREDICTOR
# -----------------
with tab_predict:
    st.markdown("### ⚔️ Matchup Settings")
    
    # Selection layouts
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox(
            "Select Team 1 (Home Team / Standard Name)",
            unique_teams,
            index=unique_teams.index("Mumbai Indians") if "Mumbai Indians" in unique_teams else 0
        )
        
    with col2:
        # Default Team 2 selection (different than Team 1)
        remaining_teams = [t for t in unique_teams if t != team1]
        team2 = st.selectbox(
            "Select Team 2 (Away Team / Standard Name)",
            remaining_teams,
            index=remaining_teams.index("Chennai Super Kings") if "Chennai Super Kings" in remaining_teams else 0
        )
        
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        venue = st.selectbox("Select Venue", unique_venues)
        
    with col4:
        # Toss Winner must be either Team 1 or Team 2
        toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
        
    with col5:
        toss_decision = st.selectbox("Toss Decision", ["field", "bat"])
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run Prediction button
    if st.button("🔮 Predict Match Outcome", use_container_width=True):
        models_path = os.path.join(project_dir, 'models')
        
        # Load correct pickel file names based on sidebar choice
        if model_choice == "Random Forest Classifier":
            model_file = "random_forest_model.pkl"
        else:
            model_file = "logistic_regression_model.pkl"
            
        preprocessor_file = "preprocessor_pipeline.pkl"
        
        preprocessor_path = os.path.join(models_path, preprocessor_file)
        model_path = os.path.join(models_path, model_file)
        
        if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
            st.error("Model or preprocessor checkpoint files not found. Please train models first by running `main.py`!")
        else:
            # Predict
            try:
                # Load preprocessor & model
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Resolve City based on Venue
                from predict import VENUE_CITY_MAP
                city = VENUE_CITY_MAP.get(venue, 'Unknown')
                
                # Build Input dataframe
                input_df = pd.DataFrame([{
                    'team1': team1,
                    'team2': team2,
                    'venue': venue,
                    'toss_winner': toss_winner,
                    'toss_decision': toss_decision,
                    'city': city
                }])
                
                # Transform using pipeline
                input_transformed = preprocessor.transform(input_df)
                
                # Get probabilities
                probabilities = model.predict_proba(input_transformed)[0]
                prob_team2_win = probabilities[0]
                prob_team1_win = probabilities[1]
                
                # Format winners
                if prob_team1_win >= prob_team2_win:
                    winner = team1
                    win_percentage = prob_team1_win * 100
                else:
                    winner = team2
                    win_percentage = prob_team2_win * 100
                
                # Visual output cards
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="winner-card">
                        <p style="margin: 0; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 2px; font-weight: bold;">Predicted Winner ({model_choice})</p>
                        <h2>🏆 {winner}</h2>
                        <p style="margin: 0; font-size: 1.2rem; font-weight: 600;">Probability: {win_percentage:.2f}%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution graphs
                col_bar1, col_bar2 = st.columns(2)
                
                with col_bar1:
                    # Team 1 Prob
                    st.markdown(f"""
                    <div class="prob-bar-label">
                        <span>{team1}</span>
                        <span>{prob_team1_win*100:.2f}%</span>
                    </div>
                    <div class="prob-bar-outer">
                        <div class="prob-bar-inner" style="width: {prob_team1_win*100}%; background: linear-gradient(90deg, #1f4068 0%, #162447 100%);"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_bar2:
                    # Team 2 Prob
                    st.markdown(f"""
                    <div class="prob-bar-label">
                        <span>{team2}</span>
                        <span>{prob_team2_win*100:.2f}%</span>
                    </div>
                    <div class="prob-bar-outer">
                        <div class="prob-bar-inner" style="width: {prob_team2_win*100}%; background: linear-gradient(90deg, #b83b5e 0%, #6a2c70 100%);"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Details Alert
                st.info(f"Match context resolved: **City** is {city}. Toss decision **{toss_decision}** by **{toss_winner}** processed into the feature pipeline.")
                
            except Exception as e:
                st.error(f"Prediction process encountered an error: {e}")

# -----------------
# TAB 2: HISTORICAL INSIGHTS
# -----------------
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis & Trends")
    st.write("These visualizations represent data analysis performed across all IPL seasons from 2008 to 2019.")
    
    outputs_path = os.path.join(project_dir, 'outputs')
    
    # In case plots aren't generated yet
    if not os.path.exists(outputs_path):
        st.warning("Visualizations output folder not found. Please execute the pipeline to generate plots.")
    else:
        # Layout plots in neat expanders/cards
        with st.expander("🥇 Team Win Percentages & Match Densities", expanded=True):
            col_eda1, col_eda2 = st.columns(2)
            with col_eda1:
                plot1_path = os.path.join(outputs_path, 'eda_win_percentage.png')
                if os.path.exists(plot1_path):
                    st.image(Image.open(plot1_path), caption="IPL Franchise Win Percentages (2008-2019)")
            with col_eda2:
                plot2_path = os.path.join(outputs_path, 'eda_matches_per_team.png')
                if os.path.exists(plot2_path):
                    st.image(Image.open(plot2_path), caption="Total Matches Played per Team")
                    
        with st.expander("🪙 Toss Impacts & Venue Performance"):
            col_eda3, col_eda4 = st.columns(2)
            with col_eda3:
                plot3_path = os.path.join(outputs_path, 'eda_toss_vs_match_winner.png')
                if os.path.exists(plot3_path):
                    st.image(Image.open(plot3_path), caption="Does Winning the Toss Help Win the Match?")
            with col_eda4:
                plot4_path = os.path.join(outputs_path, 'eda_venue_performance.png')
                if os.path.exists(plot4_path):
                    st.image(Image.open(plot4_path), caption="Top 15 Most Active Stadiums")
                    
        with st.expander("📈 Match Volume Season-by-Season"):
            plot5_path = os.path.join(outputs_path, 'eda_season_trends.png')
            if os.path.exists(plot5_path):
                st.image(Image.open(plot5_path), caption="IPL Match Count Trends", use_container_width=True)

# -----------------
# TAB 3: METRICS
# -----------------
with tab_metrics:
    st.markdown("### ⚙️ Machine Learning Models Performance & Diagnostics")
    st.write("We trained two models: Baseline Logistic Regression and Random Forest Ensemble.")
    
    outputs_path = os.path.join(project_dir, 'outputs')
    comparison_csv_path = os.path.join(outputs_path, 'model_comparison.csv')
    
    if os.path.exists(comparison_csv_path):
        st.markdown("#### 📋 Test Set Evaluation Metrics Comparison")
        metrics_df = pd.read_csv(comparison_csv_path, index_col=0)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#2c5e3b'), use_container_width=True)
    else:
        st.info("Metrics comparison table is not compiled yet. Run `main.py` first.")
        
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.markdown("#### 🔍 Feature Importances (Random Forest)")
        importance_path = os.path.join(outputs_path, 'feature_importance.png')
        if os.path.exists(importance_path):
            st.image(Image.open(importance_path), caption="Features influencing predictions the most")
            
    with col_eval2:
        st.markdown("#### 🧮 Confusion Matrices comparison")
        cm_path = os.path.join(outputs_path, 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Comparing predicted vs true values on test data")
