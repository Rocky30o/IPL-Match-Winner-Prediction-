# src/eda.py
"""
Exploratory Data Analysis (EDA) Script
-------------------------------------
This script creates and saves visualizations for:
1. Total matches played by each team.
2. Team win percentages.
3. Toss winner vs match winner analysis.
4. Venue-wise performance (distribution of matches).
5. Season-wise match trends (number of matches played per season).

Visualizations are saved as PNG files under the 'outputs/' directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df, outputs_dir):
    """Performs EDA and generates visualizations."""
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Set style for charts
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    print("\n=== Running Exploratory Data Analysis (EDA) ===")
    
    # 1. Total matches played by each team
    # A team's total matches = count in team1 + count in team2
    all_teams = pd.concat([df['team1'], df['team2']])
    matches_played = all_teams.value_counts()
    
    # 2. Team win percentages
    wins_count = df['winner'].value_counts()
    
    # Combine matches played and wins to calculate win percentage
    team_stats = pd.DataFrame({
        'Matches Played': matches_played,
        'Wins': wins_count
    }).fillna(0)
    
    team_stats['Win Percentage'] = (team_stats['Wins'] / team_stats['Matches Played']) * 100
    team_stats = team_stats.sort_values(by='Win Percentage', ascending=False)
    
    # Print statistics in terminal
    print("\n[+] Team Statistics (sorted by Win Percentage):")
    print(team_stats.round(2))
    
    # Save Plot 1: Total matches played
    plt.figure(figsize=(10, 6))
    sns.barplot(x=matches_played.values, y=matches_played.index, palette="viridis", hue=matches_played.index, legend=False)
    plt.title("Total Matches Played by Each Team (2008-2019)")
    plt.xlabel("Matches Played")
    plt.ylabel("Team")
    plt.tight_layout()
    plot_1_path = os.path.join(outputs_dir, 'eda_matches_per_team.png')
    plt.savefig(plot_1_path, dpi=150)
    plt.close()
    print(f"[+] Saved matches per team plot to: {plot_1_path}")
    
    # Save Plot 2: Win Percentage
    plt.figure(figsize=(10, 6))
    sns.barplot(x=team_stats['Win Percentage'], y=team_stats.index, palette="magma", hue=team_stats.index, legend=False)
    plt.title("IPL Team Win Percentages (2008-2019)")
    plt.xlabel("Win Percentage (%)")
    plt.ylabel("Team")
    plt.tight_layout()
    plot_2_path = os.path.join(outputs_dir, 'eda_win_percentage.png')
    plt.savefig(plot_2_path, dpi=150)
    plt.close()
    print(f"[+] Saved win percentage plot to: {plot_2_path}")
    
    # Save Plot 3: Toss Winner vs Match Winner Analysis
    # Create a boolean column: True if toss winner won the match, else False
    df['toss_winner_won'] = df['toss_winner'] == df['winner']
    toss_impact = df['toss_winner_won'].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(6, 6))
    colors = ['#4e79a7', '#f28e2b']
    plt.pie(toss_impact, labels=['Toss Winner Won', 'Toss Winner Lost'], autopct='%1.1f%%', 
            startangle=90, colors=colors, explode=(0.05, 0), shadow=True,
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title("Impact of Toss on Match Outcome", fontsize=14, weight='bold')
    plt.tight_layout()
    plot_3_path = os.path.join(outputs_dir, 'eda_toss_vs_match_winner.png')
    plt.savefig(plot_3_path, dpi=150)
    plt.close()
    print(f"[+] Saved toss vs match winner plot to: {plot_3_path}")
    
    # Save Plot 4: Venue-wise performance (top 15 venues by number of matches)
    venue_counts = df['venue'].value_counts().head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=venue_counts.values, y=venue_counts.index, palette="rocket", hue=venue_counts.index, legend=False)
    plt.title("Top 15 IPL Venues by Number of Matches Played")
    plt.xlabel("Matches Played")
    plt.ylabel("Venue")
    plt.tight_layout()
    plot_4_path = os.path.join(outputs_dir, 'eda_venue_performance.png')
    plt.savefig(plot_4_path, dpi=150)
    plt.close()
    print(f"[+] Saved venue performance plot to: {plot_4_path}")
    
    # Save Plot 5: Season-wise match trends
    season_trends = df['season'].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=season_trends.index, y=season_trends.values, marker='o', linewidth=2.5, color='#e15759')
    plt.title("IPL Matches Played Season by Season (2008-2019)")
    plt.xlabel("Season")
    plt.ylabel("Number of Matches")
    plt.xticks(season_trends.index)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_5_path = os.path.join(outputs_dir, 'eda_season_trends.png')
    plt.savefig(plot_5_path, dpi=150)
    plt.close()
    print(f"[+] Saved season trends plot to: {plot_5_path}")
    
    print("[+] All EDA visualizations generated successfully.")

if __name__ == "__main__":
    # Test run
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import load_data, clean_data
    
    data_path = os.path.join(project_dir, 'data', 'matches.csv')
    out_dir = os.path.join(project_dir, 'outputs')
    try:
        raw_df = load_data(data_path)
        cleaned_df = clean_data(raw_df)
        run_eda(cleaned_df, out_dir)
    except Exception as e:
        print(f"[-] Error running EDA: {e}")
