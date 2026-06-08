# src/generate_notebook.py
"""
Jupyter Notebook Generator
--------------------------
Since we cannot modify .ipynb files directly in some environments, this script
programmatically generates the Jupyter Notebook 'notebook/ipl_match_winner_prediction.ipynb'
with clean markdown explanations and code cells that replicate the entire ML pipeline.

This notebook is designed to be highly educational and suitable for student interviews.
"""

import os
import json

def generate_notebook():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notebook_dir = os.path.join(project_dir, 'notebook')
    os.makedirs(notebook_dir, exist_ok=True)
    
    notebook_path = os.path.join(notebook_dir, 'ipl_match_winner_prediction.ipynb')
    
    print(f"[*] Generating Jupyter Notebook at: {notebook_path}")
    
    # Define notebook cells
    cells = []
    
    # 1. Introduction Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# IPL Match Winner Prediction\n",
            "### An End-to-End Machine Learning Project\n",
            "\n",
            "This Jupyter Notebook demonstrates a complete end-to-end Machine Learning pipeline to predict the winner of an Indian Premier League (IPL) match using historical match data. \n",
            "\n",
            "### Objectives:\n",
            "1. **Data Collection & Ingestion**: Load the IPL Matches dataset and clean missing values.\n",
            "2. **Exploratory Data Analysis (EDA)**: Understand historical match data, win percentages, toss impacts, and seasonal trends.\n",
            "3. **Data Preprocessing**: Structure features and encode categorical columns using a robust scikit-learn preprocessing pipeline.\n",
            "4. **Model Training**: Fit and compare a Logistic Regression baseline and an ensemble Random Forest Classifier.\n",
            "5. **Model Evaluation**: Evaluate models using Accuracy, Precision, Recall, F1 Score, and Confusion Matrices.\n",
            "6. **Feature Importance**: Analyze what factors influence match outcomes the most.\n",
            "7. **Prediction System**: Make predictions on future/unseen matches and display winning probabilities.\n",
            "\n",
            "Let's get started!"
        ]
    })
    
    # 2. Imports
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import all necessary libraries\n",
            "import os\n",
            "import urllib.request\n",
            "import pickle\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Scikit-Learn tools\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n",
            "\n",
            "# Set plot styles\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams.update({'figure.figsize': (10, 6), 'font.size': 11})\n",
            "print(\"[+] Libraries successfully imported!\")"
        ]
    })
    
    # 3. Data Collection Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Collection & Ingestion\n",
            "\n",
            "We will download the historical IPL matches dataset (covering seasons 2008-2019) from a raw GitHub repository and load it into a Pandas DataFrame."
        ]
    })
    
    # 4. Download and Load Data Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Download the dataset\n",
            "DATA_URL = \"https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/matches.csv\"\n",
            "data_dir = \"../data\"\n",
            "os.makedirs(data_dir, exist_ok=True)\n",
            "csv_path = os.path.join(data_dir, \"matches.csv\")\n",
            "\n",
            "if not os.path.exists(csv_path):\n",
            "    print(\"[*] Downloading matches.csv from GitHub...\")\n",
            "    urllib.request.urlretrieve(DATA_URL, csv_path)\n",
            "    print(\"[+] Download complete!\")\n",
            "else:\n",
            "    print(\"[+] Dataset file already exists locally.\")\n",
            "\n",
            "# Load using Pandas\n",
            "df_raw = pd.read_csv(csv_path)\n",
            "print(f\"\\n[+] Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns\")\n",
            "df_raw.head()"
        ]
    })
    
    # 5. Data Cleaning Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Cleaning & Standardization\n",
            "\n",
            "To build an accurate prediction model, we must clean our data:\n",
            "1. **Standardize Team Names**: Over the seasons, some team names changed (e.g., Delhi Daredevils renamed to Delhi Capitals, or Kings XI Punjab to Punjab Kings). Standardizing these ensures our model treats them as the same team.\n",
            "2. **Handle Missing Values**: Impute missing city names based on their venues, and remove matches with no winner (e.g., abandoned due to rain).\n",
            "3. **Drop Irrelevant Features**: Drop features that are only available *after* the match has finished (e.g., umpires, player of the match, winning margin, duckworth-lewis applied, etc.) since we cannot use them for predicting upcoming matches."
        ]
    })
    
    # 6. Cleaning Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = df_raw.copy()\n",
            "\n",
            "# 1. Standardize team names\n",
            "team_mapping = {\n",
            "    'Rising Pune Supergiants': 'Rising Pune Supergiant',\n",
            "    'Delhi Daredevils': 'Delhi Capitals',\n",
            "    'Kings XI Punjab': 'Punjab Kings',\n",
            "    'Deccan Chargers': 'Sunrisers Hyderabad'\n",
            "}\n",
            "for col in ['team1', 'team2', 'toss_winner', 'winner']:\n",
            "    df[col] = df[col].replace(team_mapping)\n",
            "\n",
            "# 2. Drop matches with no winner (abandoned/ties with no superover winner)\n",
            "df = df.dropna(subset=['winner'])\n",
            "\n",
            "# 3. Fill missing cities using venue names\n",
            "venue_city_map = {\n",
            "    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',\n",
            "    'Maharashtra Cricket Association Stadium': 'Pune',\n",
            "    'Saurashtra Cricket Association Stadium': 'Rajkot',\n",
            "    'Holkar Cricket Stadium': 'Indore',\n",
            "    'M Chinnaswamy Stadium': 'Bengaluru',\n",
            "    'Wankhede Stadium': 'Mumbai',\n",
            "    'Eden Gardens': 'Kolkata',\n",
            "    'Feroz Shah Kotla': 'Delhi',\n",
            "    'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Mohali',\n",
            "    'Green Park': 'Kanpur',\n",
            "    'Punjab Cricket Association Stadium, Mohali': 'Mohali',\n",
            "    'Sawai Mansingh Stadium': 'Jaipur',\n",
            "    'JSCA International Stadium Complex': 'Ranchi',\n",
            "    'Brabourne Stadium': 'Mumbai',\n",
            "    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Visakhapatnam',\n",
            "    'Subrata Roy Sahara Stadium': 'Pune',\n",
            "    'Dr DY Patil Sports Academy': 'Mumbai',\n",
            "    'New Kingsmead Stadium': 'Durban',\n",
            "    'St George\\'s Park': 'Port Elizabeth',\n",
            "    'Kingsmead': 'Durban',\n",
            "    'SuperSport Park': 'Centurion',\n",
            "    'Buffalo Park': 'East London',\n",
            "    'Newlands': 'Cape Town',\n",
            "    'De Beers Diamond Oval': 'Kimberley',\n",
            "    'OUTsurance Oval': 'Bloemfontein',\n",
            "    'Sharjah Cricket Stadium': 'Sharjah',\n",
            "    'Dubai International Cricket Stadium': 'Dubai',\n",
            "    'Sheikh Zayed Stadium': 'Abu Dhabi'\n",
            "}\n",
            "missing_city_mask = df['city'].isnull()\n",
            "df.loc[missing_city_mask, 'city'] = df.loc[missing_city_mask, 'venue'].map(venue_city_map)\n",
            "df['city'] = df['city'].fillna('Unknown')\n",
            "\n",
            "# 4. Drop irrelevant columns\n",
            "columns_to_drop = ['id', 'date', 'result', 'dl_applied', 'win_by_runs', \n",
            "                   'win_by_wickets', 'player_of_match', 'umpire1', 'umpire2', 'umpire3']\n",
            "df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')\n",
            "\n",
            "print(f\"[+] Cleaned Dataset Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns\")\n",
            "print(f\"[+] Cleaned columns: {list(df_cleaned.columns)}\")\n",
            "df_cleaned.head()"
        ]
    })
    
    # 7. EDA Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA)\n",
            "\n",
            "Let's explore key statistics and visualize match trends to build domain knowledge."
        ]
    })
    
    # 8. EDA Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Total matches played by each team\n",
            "all_teams = pd.concat([df_cleaned['team1'], df_cleaned['team2']])\n",
            "matches_played = all_teams.value_counts()\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(x=matches_played.values, y=matches_played.index, palette=\"viridis\", hue=matches_played.index, legend=False)\n",
            "plt.title(\"Total Matches Played by Each Team (2008-2019)\")\n",
            "plt.xlabel(\"Matches Played\")\n",
            "plt.ylabel(\"Team\")\n",
            "plt.show()\n",
            "\n",
            "# 2. Team Win Percentages\n",
            "wins_count = df_cleaned['winner'].value_counts()\n",
            "team_stats = pd.DataFrame({'Matches': matches_played, 'Wins': wins_count}).fillna(0)\n",
            "team_stats['Win %'] = (team_stats['Wins'] / team_stats['Matches']) * 100\n",
            "team_stats = team_stats.sort_values(by='Win %', ascending=False)\n",
            "\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(x=team_stats['Win %'], y=team_stats.index, palette=\"magma\", hue=team_stats.index, legend=False)\n",
            "plt.title(\"IPL Team Win Percentages (2008-2019)\")\n",
            "plt.xlabel(\"Win %\")\n",
            "plt.ylabel(\"Team\")\n",
            "plt.show()\n",
            "print(team_stats.round(2))"
        ]
    })
    
    # 9. Toss impact code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Toss winner vs Match winner analysis\n",
            "toss_match_win = (df_cleaned['toss_winner'] == df_cleaned['winner']).value_counts(normalize=True) * 100\n",
            "\n",
            "plt.figure(figsize=(5, 5))\n",
            "plt.pie(toss_match_win, labels=['Toss Winner Won', 'Toss Winner Lost'], autopct='%1.1f%%', \n",
            "        colors=['#4e79a7', '#f28e2b'], startangle=90, explode=(0.05, 0), shadow=True)\n",
            "plt.title(\"Toss Winner vs Match Winner Impact\", fontsize=13, weight='bold')\n",
            "plt.show()"
        ]
    })
    
    # 10. Venue and seasons
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Top 15 venues by number of matches\n",
            "venue_counts = df_cleaned['venue'].value_counts().head(15)\n",
            "plt.figure(figsize=(10, 5))\n",
            "sns.barplot(x=venue_counts.values, y=venue_counts.index, palette=\"rocket\", hue=venue_counts.index, legend=False)\n",
            "plt.title(\"Top 15 IPL Venues by Number of Matches Played\")\n",
            "plt.xlabel(\"Matches Played\")\n",
            "plt.ylabel(\"Venue\")\n",
            "plt.show()\n",
            "\n",
            "# 5. Season-wise match trends\n",
            "season_trends = df_cleaned['season'].value_counts().sort_index()\n",
            "plt.figure(figsize=(10, 4))\n",
            "sns.lineplot(x=season_trends.index, y=season_trends.values, marker='o', linewidth=2.5, color='#e15759')\n",
            "plt.title(\"IPL Matches Played Season by Season (2008-2019)\")\n",
            "plt.xlabel(\"Season\")\n",
            "plt.ylabel(\"Matches Played\")\n",
            "plt.xticks(season_trends.index)\n",
            "plt.show()"
        ]
    })
    
    # 11. Preprocessing Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Data Preprocessing\n",
            "\n",
            "To feed this data into a Machine Learning model, we must structure and transform it:\n",
            "1. **Formulate Target**: Since predicting the winner out of all 15+ teams can lead to predicting a team that isn't playing, we structure this as binary classification: **Does Team 1 win (Target = 1) or Team 2 win (Target = 0)?**\n",
            "2. **Split Features and Target**: Extract features (`team1`, `team2`, `venue`, `toss_winner`, `toss_decision`, `city`) and target (`target`).\n",
            "3. **Train-Test Split**: Split dataset into **80% Training** (to train models) and **20% Testing** (to evaluate models).\n",
            "4. **One-Hot Encoding**: Use scikit-learn's `ColumnTransformer` and `OneHotEncoder` to convert categorical text values into binary numerical arrays."
        ]
    })
    
    # 12. Preprocessing Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Filter and create binary target\n",
            "df_model = df_cleaned[(df_cleaned['winner'] == df_cleaned['team1']) | (df_cleaned['winner'] == df_cleaned['team2'])].copy()\n",
            "df_model['target'] = (df_model['winner'] == df_model['team1']).astype(int)\n",
            "\n",
            "# 2. Extract Features and Target\n",
            "feature_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']\n",
            "X = df_model[feature_cols]\n",
            "y = df_model['target']\n",
            "\n",
            "# 3. Train-Test Split (80:20)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "print(f\"[+] Training set size: {X_train.shape[0]}\")\n",
            "print(f\"[+] Testing set size: {X_test.shape[0]}\")\n",
            "\n",
            "# 4. Preprocessing Pipeline using ColumnTransformer and OneHotEncoder\n",
            "categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)\n",
            "    ], \n",
            "    remainder='drop'\n",
            ")\n",
            "\n",
            "preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])\n",
            "\n",
            "# Fit pipeline on training data\n",
            "preprocessing_pipeline.fit(X_train)\n",
            "print(\"[+] Preprocessing pipeline successfully defined and fitted!\")\n",
            "\n",
            "# Transform features for verification\n",
            "X_train_transformed = preprocessing_pipeline.transform(X_train)\n",
            "print(f\"[+] Transformed training shape: {X_train_transformed.shape}\")"
        ]
    })
    
    # 13. Training Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Machine Learning Models\n",
            "\n",
            "We will train and compare two classification algorithms:\n",
            "1. **Logistic Regression**: A linear model that estimates the probability of binary outcomes. It serves as our baseline.\n",
            "2. **Random Forest Classifier**: An ensemble of decision trees that builds robust, non-linear boundaries. It is well-suited for high-dimensional categorical features."
        ]
    })
    
    # 14. Training Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Transform training and testing features\n",
            "X_train_trans = preprocessing_pipeline.transform(X_train)\n",
            "X_test_trans = preprocessing_pipeline.transform(X_test)\n",
            "\n",
            "# 1. Train Logistic Regression Classifier\n",
            "lr_model = LogisticRegression(max_iter=1000, random_state=42)\n",
            "lr_model.fit(X_train_trans, y_train)\n",
            "print(\"[+] Logistic Regression trained successfully!\")\n",
            "\n",
            "# 2. Train Random Forest Classifier\n",
            "rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
            "rf_model.fit(X_train_trans, y_train)\n",
            "print(\"[+] Random Forest Classifier trained successfully!\")"
        ]
    })
    
    # 15. Evaluation Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Model Evaluation & Comparison\n",
            "\n",
            "We will evaluate both models on our unseen test dataset using **Accuracy**, **Precision**, **Recall**, and **F1 Score** metrics, and draw **Confusion Matrices**."
        ]
    })
    
    # 16. Evaluation Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Predictions\n",
            "y_pred_lr = lr_model.predict(X_test_trans)\n",
            "y_pred_rf = rf_model.predict(X_test_trans)\n",
            "\n",
            "# Define metrics dictionary\n",
            "metrics = {\n",
            "    'Logistic Regression': {\n",
            "        'Accuracy': accuracy_score(y_test, y_pred_lr),\n",
            "        'Precision': precision_score(y_test, y_pred_lr),\n",
            "        'Recall': recall_score(y_test, y_pred_lr),\n",
            "        'F1 Score': f1_score(y_test, y_pred_lr)\n",
            "    },\n",
            "    'Random Forest': {\n",
            "        'Accuracy': accuracy_score(y_test, y_pred_rf),\n",
            "        'Precision': precision_score(y_test, y_pred_rf),\n",
            "        'Recall': recall_score(y_test, y_pred_rf),\n",
            "        'F1 Score': f1_score(y_test, y_pred_rf)\n",
            "    }\n",
            "}\n",
            "\n",
            "# Display metrics table\n",
            "comparison_df = pd.DataFrame(metrics).T\n",
            "print(\"=== Model Comparison ===\")\n",
            "print(comparison_df.round(4))\n",
            "\n",
            "# Plot Confusion Matrices\n",
            "cm_lr = confusion_matrix(y_test, y_pred_lr)\n",
            "cm_rf = confusion_matrix(y_test, y_pred_rf)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "# Logistic Regression CM heatmap\n",
            "sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],\n",
            "            xticklabels=['Team 2 Wins', 'Team 1 Wins'], yticklabels=['Team 2 Wins', 'Team 1 Wins'])\n",
            "axes[0].set_title(\"Logistic Regression Confusion Matrix\")\n",
            "axes[0].set_xlabel(\"Predicted\")\n",
            "axes[0].set_ylabel(\"True\")\n",
            "\n",
            "# Random Forest CM heatmap\n",
            "sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],\n",
            "            xticklabels=['Team 2 Wins', 'Team 1 Wins'], yticklabels=['Team 2 Wins', 'Team 1 Wins'])\n",
            "axes[1].set_title(\"Random Forest Confusion Matrix\")\n",
            "axes[1].set_xlabel(\"Predicted\")\n",
            "axes[1].set_ylabel(\"True\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # 17. Feature Importance Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Feature Importance\n",
            "\n",
            "Let's extract the top features that influence match outcomes in the Random Forest model."
        ]
    })
    
    # 18. Feature Importance Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Extract feature names from OHE\n",
            "feature_names = preprocessing_pipeline.named_steps['preprocessor'].get_feature_names_out()\n",
            "clean_feature_names = [name.replace('cat__', '') for name in feature_names]\n",
            "\n",
            "# Get importances\n",
            "importances = rf_model.feature_importances_\n",
            "importance_df = pd.DataFrame({\n",
            "    'Feature': clean_feature_names,\n",
            "    'Importance': importances\n",
            "}).sort_values(by='Importance', ascending=False)\n",
            "\n",
            "# Plot top 15 features\n",
            "plt.figure(figsize=(10, 6))\n",
            "top_15 = importance_df.head(15)\n",
            "sns.barplot(x=top_15['Importance'], y=top_15['Feature'], palette=\"crest\", hue=top_15['Feature'], legend=False)\n",
            "plt.title(\"Top 15 Feature Importances (Random Forest)\")\n",
            "plt.xlabel(\"Importance\")\n",
            "plt.ylabel(\"Encoded Feature\")\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Show list\n",
            "print(\"[+] Top 10 Features list:\")\n",
            "print(importance_df.head(10).to_string(index=False))"
        ]
    })
    
    # 19. Prediction System Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Prediction System\n",
            "\n",
            "Now let's build the interactive prediction function. Users can input any upcoming matchup details and retrieve the predicted winner and respective probabilities."
        ]
    })
    
    # 20. Prediction System Code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Venue to City mapping dictionary (copied from earlier processing)\n",
            "VENUE_MAP = {\n",
            "    'Wankhede Stadium': 'Mumbai',\n",
            "    'Eden Gardens': 'Kolkata',\n",
            "    'M Chinnaswamy Stadium': 'Bengaluru',\n",
            "    'Feroz Shah Kotla': 'Delhi',\n",
            "    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',\n",
            "    'Sawai Mansingh Stadium': 'Jaipur',\n",
            "    'Punjab Cricket Association Stadium, Mohali': 'Mohali'\n",
            "}\n",
            "\n",
            "def predict_winner(team1, team2, venue, toss_winner, toss_decision):\n",
            "    \"\"\"\n",
            "    Helper prediction function\n",
            "    \"\"\"\n",
            "    city = VENUE_MAP.get(venue, 'Unknown')\n",
            "    \n",
            "    # Prepare input frame\n",
            "    match_df = pd.DataFrame([{\n",
            "        'team1': team1,\n",
            "        'team2': team2,\n",
            "        'venue': venue,\n",
            "        'toss_winner': toss_winner,\n",
            "        'toss_decision': toss_decision,\n",
            "        'city': city\n",
            "    }])\n",
            "    \n",
            "    # Transform features using preprocessor pipeline\n",
            "    match_trans = preprocessing_pipeline.transform(match_df)\n",
            "    \n",
            "    # Predict probabilities\n",
            "    proba = rf_model.predict_proba(match_trans)[0]\n",
            "    prob_team2 = proba[0]\n",
            "    prob_team1 = proba[1]\n",
            "    \n",
            "    # Output result\n",
            "    print(\"=\"*50)\n",
            "    print(\"             IPL MATCH WINNER PREDICTOR         \")\n",
            "    print(\"=\"*50)\n",
            "    print(f\"{team1} vs {team2}\")\n",
            "    print(f\"Venue: {venue} | Toss Winner: {toss_winner} | Toss Decision: {toss_decision}\")\n",
            "    print(\"-\"*50)\n",
            "    \n",
            "    if prob_team1 >= prob_team2:\n",
            "        print(f\"Predicted Winner:    **{team1}**\")\n",
            "        print(f\"Winning Probability: {prob_team1 * 100:.2f}%\")\n",
            "    else: \n",
            "        print(f\"Predicted Winner:    **{team2}**\")\n",
            "        print(f\"Winning Probability: {prob_team2 * 100:.2f}%\")\n",
            "    print(\"-\"*50)\n",
            "    print(f\"Probability distribution:\")\n",
            "    print(f\"   - {team1}: {prob_team1*100:.2f}%\")\n",
            "    print(f\"   - {team2}: {prob_team2*100:.2f}%\")\n",
            "    print(\"=\"*50)\n",
            "\n",
            "# Run a Sample Prediction\n",
            "predict_winner(\n",
            "    team1=\"Mumbai Indians\",\n",
            "    team2=\"Chennai Super Kings\",\n",
            "    venue=\"Wankhede Stadium\",\n",
            "    toss_winner=\"Mumbai Indians\",\n",
            "    toss_decision=\"field\"\n",
            ")"
        ]
    })
    
    # Notebook Metadata
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Save file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1)
        
    print(f"[+] Jupyter Notebook successfully generated at: {notebook_path}")

if __name__ == "__main__":
    generate_notebook()
