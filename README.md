# IPL Match Winner Prediction 🏏🔮

An end-to-end Machine Learning project to predict the winner of an Indian Premier League (IPL) match using historical match data. Built with clean, well-documented code designed for students preparing for data science interviews.

---

## 📌 Project Overview
The Indian Premier League (IPL) is one of the most popular and unpredictable T20 cricket leagues in the world. Predicting the winner of a match involves complex variables. This project builds a binary classification model to predict which of the two playing teams will win a match, based on historical match-level metadata (seasons 2008-2019).

Rather than predicting a winning team from a list of all franchises—which could lead to logical errors (e.g., predicting Royal Challengers Bangalore to win a match between Mumbai Indians and Chennai Super Kings)—**this system models the match as a binary classification problem**:
* **Class 1**: Team 1 Wins
* **Class 0**: Team 2 Wins

---

## 📊 Dataset Information
The project utilizes the classic **IPL Matches Dataset** containing match summaries from 2008 to 2019.
Key features used in training:
* `team1`: Name of Team 1
* `team2`: Name of Team 2
* `venue`: Stadium name where the match is played
* `city`: City where the match is played
* `toss_winner`: The team that won the toss
* `toss_decision`: The choice made by the toss winner (`bat` or `field`)
* `season`: Year of the tournament

---

## 🛠️ Technologies Used
* **Python 3.12**
* **Pandas**: Data manipulation and cleaning
* **NumPy**: Numerical operations
* **Scikit-Learn**: Preprocessing pipeline (`ColumnTransformer`, `OneHotEncoder`, `Pipeline`), split (`train_test_split`), and classification models (`LogisticRegression`, `RandomForestClassifier`)
* **Matplotlib & Seaborn**: Scientific visualizations
* **Jupyter Notebook**: Interactive learning and demonstration

---

## 📂 Project Structure
```
IPL-Match-Winner-Prediction/
│
├── data/
│   └── matches.csv                     # Raw downloaded dataset
│
├── notebook/
│   └── ipl_match_winner_prediction.ipynb # Fully documented, runnable Jupyter notebook
│
├── src/
│   ├── __init__.py                     # Package marker
│   ├── download_data.py                # Automates dataset retrieval
│   ├── data_ingestion.py               # Loads data, standardizes team names, handles missing values
│   ├── eda.py                          # Generates and saves EDA plots
│   ├── preprocessing.py                # Formulates target, fits and saves preprocessor pipeline
│   ├── model_training.py               # Fits Logistic Regression & Random Forest classifiers
│   ├── evaluation.py                   # Evaluates models, saves confusion matrix & feature importance
│   ├── predict.py                      # Match inference interface with probability outputs
│   └── generate_notebook.py            # Generates the Jupyter Notebook programmatically
│
├── models/
│   ├── preprocessor_pipeline.pkl       # Saved scikit-learn OneHotEncoder pipeline
│   ├── logistic_regression_model.pkl   # Saved Logistic Regression classifier
│   └── random_forest_model.pkl         # Saved Random Forest classifier
│
├── outputs/
│   ├── eda_matches_per_team.png        # Bar chart: Matches played per team
│   ├── eda_win_percentage.png          # Bar chart: Historical win percentages
│   ├── eda_toss_vs_match_winner.png    # Pie chart: Toss winner correlation with match winner
│   ├── eda_venue_performance.png       # Bar chart: Distribution of matches across top 15 venues
│   ├── eda_season_trends.png           # Line chart: Matches played per season
│   ├── confusion_matrices.png          # Confusion matrices for both trained models
│   ├── feature_importance.png          # Random Forest feature importance chart
│   └── model_comparison.csv            # Metrics CSV comparing accuracy, precision, recall, F1
│
├── main.py                             # Driver script running the entire pipeline end-to-end
├── requirements.txt                    # Project requirements file
└── README.md                           # Professional README documentation
```

---

## ⚙️ Setup & Execution

### 1. Prerequisites
Ensure you have Python installed. Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
To download the data, perform EDA, train both models, print predictions, and compile the Jupyter Notebook, run:
```bash
python main.py
```

### 3. Start the Notebook
Open the generated notebook to review the step-by-step breakdown:
```bash
jupyter notebook notebook/ipl_match_winner_prediction.ipynb
```

---

## 📈 Methodology & Pipeline Steps

### 1. Data Cleaning & Standardization
* **Franchise Name Mapping**: Standardized historical name changes (e.g., *Delhi Daredevils* to *Delhi Capitals*, *Kings XI Punjab* to *Punjab Kings*) to prevent duplicated features.
* **Missing Value Imputation**: Automatically resolved missing `city` fields using a custom map of `venue` names.
* **Post-Match Feature Removal**: Dropped columns like `win_by_runs`, `win_by_wickets`, and `player_of_match` that are only known after a game ends.

### 2. Preprocessing
We created a scikit-learn preprocessing `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')`. This encodes the categorical string features into numeric arrays, making it safe to run prediction even with brand new venues or teams.

### 3. Model Comparison
We trained a baseline **Logistic Regression** and a **Random Forest Classifier** using a stratified 80:20 train-test split.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 49.67% | 41.82% | 34.33% | 37.70% |
| **Random Forest** | 47.02% | 39.68% | 37.31% | 38.46% |

> [!WARNING]
> **Why is the accuracy close to 50% (random guess)?**
> This is a crucial concept to explain during interviews!
> Match outcome predictions based *only* on team names and venue are inherently limited. In T20 cricket, players change franchises during annual auctions, meaning "Mumbai Indians" in 2008 had an entirely different squad compared to 2019. Basic models cannot capture this. To improve prediction accuracy, features must incorporate **player-level statistics** (e.g., current batsman strike rate, bowler economy, head-to-head match-ups, and live weather conditions).

---

## 📊 Feature Importance
The top features influencing the Random Forest model:
1. `team1_Chennai Super Kings` (High historical win percentage)
2. `team2_Punjab Kings`
3. `team1_Punjab Kings`
4. `team1_Mumbai Indians`
5. `team2_Kolkata Knight Riders`

---

## 🔮 Prediction Interface
The inference module `src/predict.py` takes a matchup scenario, automatically maps the city from the venue, transforms the variables, and runs the classifier.
Example output:
```
==================================================
           IPL MATCH WINNER PREDICTION          
==================================================
Matchup:       Mumbai Indians vs Chennai Super Kings
Venue:         Wankhede Stadium (City: Mumbai)
Toss Winner:   Mumbai Indians (field first)
--------------------------------------------------
Predicted Winner:    Mumbai Indians
Winning Probability: 65.18%
--------------------------------------------------
Win Probability breakdown:
   - Mumbai Indians: 65.18%
   - Chennai Super Kings: 34.82%
==================================================
```

---

## 🚀 Future Improvements
1. **Incorporate Player Stats**: Add batsmen and bowler ratings, recent form, and average match-up scores.
2. **Incorporate Live Match Data**: Build a model that predicts match outcomes ball-by-ball using live runs, wickets, and overs.
3. **Hyperparameter Tuning**: Run Grid Search or Random Search on Random Forest estimators to optimize performance.
4. **Deploy as Web App**: Build a Streamlit or Flask web dashboard for user inputs.

---

## 💼 Resume Points (Data Science Intern)
Add these ATS-friendly points to your resume:

* **End-to-End ML Pipeline**: Architected an end-to-end IPL Match Winner Prediction pipeline in Python, automating data collection, ingestion, exploratory data analysis, preprocessing, and model inference.
* **Robust Preprocessing & Target Modeling**: Formulated the target variable as a binary classification (Team 1 vs. Team 2) to eliminate prediction errors, and built a reusable scikit-learn preprocessing pipeline utilizing `ColumnTransformer` and `OneHotEncoder`.
* **Model Benchmarking & Evaluation**: Benchmarked Logistic Regression and Random Forest Classifiers on a stratified 80:20 split, evaluating performance with Accuracy, Precision, Recall, F1 Score, and Confusion Matrices.
* **Feature Engineering & Explainability**: Extracted Random Forest Gini importance to identify top match outcome drivers and documented domain-specific model limitations (such as player squad turnover) to explain performance during review.
