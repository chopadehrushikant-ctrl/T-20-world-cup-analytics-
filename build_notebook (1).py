import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# 🏏 T20 World Cup Data Analytics Project

**End-to-end analytics pipeline:** data loading → cleaning → EDA → feature engineering → machine learning (match-winner prediction) → summary report.

**Dataset note:** This project uses a programmatically generated dataset that mirrors the real structure of T20 World Cup data (real team names, real venue names, realistic score/toss/venue patterns modeled on actual T20I strength tiers). Match outcomes are simulated, not real historical results. To use real data instead, drop a real CSV into `data/matches.csv` with the same column names and every cell below will run unchanged — see the schema reference in Section 1.

**Tech stack:** pandas, numpy, matplotlib, seaborn, scikit-learn
"""))

cells.append(md("## 1. Setup & Data Loading"))

cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.max_columns", None)

matches = pd.read_csv("../data/matches.csv", parse_dates=["date"])
batting = pd.read_csv("../data/batting_summary.csv")
bowling = pd.read_csv("../data/bowling_summary.csv")

print("Matches:", matches.shape)
print("Batting summary:", batting.shape)
print("Bowling summary:", bowling.shape)
matches.head()
"""))

cells.append(md("""**Schema reference (`matches.csv`):**

| column | meaning |
|---|---|
| match_id | unique match identifier |
| date | match date |
| stage | Group Stage / Super 8 / Semi Final / Final |
| venue, city | ground and host city |
| team1, team2 | competing teams |
| toss_winner, toss_decision | toss result |
| team1_score, team1_wickets, team1_overs | 1st innings |
| team2_score, team2_wickets, team2_overs | 2nd innings |
| winner | match winner |
| win_margin | e.g. "23 runs" or "4 wickets" |
| player_of_match | POTM award |
"""))

cells.append(md("## 2. Data Cleaning"))

cells.append(code("""# Check for missing values and dtypes
print(matches.isna().sum())
print()
print(matches.dtypes)
"""))

cells.append(code("""# Standardize text columns, ensure numeric columns are numeric
text_cols = ["team1", "team2", "toss_winner", "winner", "venue", "city", "stage", "toss_decision", "player_of_match"]
for c in text_cols:
    matches[c] = matches[c].astype(str).str.strip()

numeric_cols = ["team1_score", "team1_wickets", "team2_score", "team2_wickets"]
for c in numeric_cols:
    matches[c] = pd.to_numeric(matches[c], errors="coerce")

matches = matches.drop_duplicates(subset="match_id").reset_index(drop=True)
print("Clean shape:", matches.shape)
"""))

cells.append(code("""# Engineer a few useful columns
matches["run_margin"] = matches["team1_score"] - matches["team2_score"]
matches["toss_winner_won_match"] = matches["toss_winner"] == matches["winner"]
matches["total_match_runs"] = matches["team1_score"] + matches["team2_score"]
matches["month"] = matches["date"].dt.month_name()
matches.head(3)
"""))

cells.append(md("## 3. Exploratory Data Analysis"))

cells.append(md("### 3.1 Wins by team"))
cells.append(code("""win_counts = matches["winner"].value_counts()

plt.figure(figsize=(11, 6))
sns.barplot(x=win_counts.values, y=win_counts.index, hue=win_counts.index, palette="viridis", legend=False)
plt.title("Total Wins by Team")
plt.xlabel("Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.2 Does winning the toss help win the match?"))
cells.append(code("""toss_impact = matches["toss_winner_won_match"].value_counts(normalize=True) * 100
print(toss_impact)

plt.figure(figsize=(5, 5))
plt.pie(toss_impact.values, labels=["Toss winner won match", "Toss winner lost match"],
        autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=90)
plt.title("Toss Impact on Match Outcome")
plt.show()
"""))

cells.append(md("### 3.3 Score distributions"))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(matches["team1_score"], bins=15, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("1st Innings Score Distribution")
sns.histplot(matches["team2_score"], bins=15, kde=True, ax=axes[1], color="#DD8452")
axes[1].set_title("2nd Innings (Chase) Score Distribution")
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.4 Venue analysis — average total runs per venue"))
cells.append(code("""venue_runs = matches.groupby("venue")["total_match_runs"].mean().sort_values(ascending=False)

plt.figure(figsize=(11, 6))
sns.barplot(x=venue_runs.values, y=venue_runs.index, hue=venue_runs.index, palette="mako", legend=False)
plt.title("Average Total Match Runs by Venue")
plt.xlabel("Avg. total runs (both innings)")
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.5 Stage-wise match count & average scores"))
cells.append(code("""stage_summary = matches.groupby("stage").agg(
    matches=("match_id", "count"),
    avg_team1_score=("team1_score", "mean"),
    avg_team2_score=("team2_score", "mean"),
).round(1)
stage_order = ["Group Stage", "Super 8", "Semi Final", "Final"]
stage_summary = stage_summary.reindex([s for s in stage_order if s in stage_summary.index])
stage_summary
"""))

cells.append(md("### 3.6 Top run-scorers and wicket-takers"))
cells.append(code("""top_batters = batting.sort_values("runs", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_batters, x="runs", y="player", hue="team", dodge=False)
plt.title("Top 10 Run Scorers")
plt.xlabel("Runs")
plt.tight_layout()
plt.show()
top_batters[["player", "team", "runs", "strike_rate", "fifties", "hundreds"]]
"""))

cells.append(code("""top_bowlers = bowling.sort_values("wickets", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_bowlers, x="wickets", y="player", hue="team", dodge=False)
plt.title("Top 10 Wicket Takers")
plt.xlabel("Wickets")
plt.tight_layout()
plt.show()
top_bowlers[["player", "team", "wickets", "economy_rate", "best_bowling"]]
"""))

cells.append(md("### 3.7 Correlation heatmap of numeric match features"))
cells.append(code("""num_cols = ["team1_score", "team1_wickets", "team2_score", "team2_wickets", "run_margin", "total_match_runs"]
plt.figure(figsize=(8, 6))
sns.heatmap(matches[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Match Features")
plt.tight_layout()
plt.show()
"""))

cells.append(md("## 4. Feature Engineering for Machine Learning\\n\\n**Goal:** predict the match winner (team1 or team2) from pre/mid-match information — team strength proxy (historical win rate), toss result, and venue."))

cells.append(code("""# Historical win-rate per team (computed excluding the current match to avoid leakage
# would require a walk-forward approach; for this project-scale dataset we use overall
# win rate as a team-strength proxy feature, which is a common simplification).
win_rate = (matches["winner"].value_counts() / (
    matches["team1"].value_counts().add(matches["team2"].value_counts(), fill_value=0)
)).fillna(0)

ml_df = matches.copy()
ml_df["team1_win_rate"] = ml_df["team1"].map(win_rate)
ml_df["team2_win_rate"] = ml_df["team2"].map(win_rate)
ml_df["toss_winner_is_team1"] = (ml_df["toss_winner"] == ml_df["team1"]).astype(int)
ml_df["toss_decision_bat"] = (ml_df["toss_decision"] == "Bat").astype(int)
ml_df["target"] = (ml_df["winner"] == ml_df["team1"]).astype(int)  # 1 = team1 wins

le_venue = LabelEncoder()
ml_df["venue_enc"] = le_venue.fit_transform(ml_df["venue"])
le_stage = LabelEncoder()
ml_df["stage_enc"] = le_stage.fit_transform(ml_df["stage"])

feature_cols = ["team1_win_rate", "team2_win_rate", "toss_winner_is_team1",
                 "toss_decision_bat", "venue_enc", "stage_enc"]
X = ml_df[feature_cols]
y = ml_df["target"]
X.head()
"""))

cells.append(md("## 5. Model Training — Match Winner Prediction"))

cells.append(code("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    results[name] = {"model": model, "accuracy": acc, "auc": auc, "preds": preds}
    print(f"{name}: accuracy={acc:.3f}, ROC-AUC={auc:.3f}")
"""))

cells.append(md("### 5.1 Confusion matrix — best model"))
cells.append(code("""best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print("Best model:", best_name)
print(classification_report(y_test, best["preds"], target_names=["team2 wins", "team1 wins"]))

cm = confusion_matrix(y_test, best["preds"])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["team2 wins", "team1 wins"], yticklabels=["team2 wins", "team1 wins"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 5.2 Feature importance (Random Forest)"))
cells.append(code("""rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="crest", legend=False)
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
importances
"""))

cells.append(md("## 6. Summary Report\\n\\nRun the cell below to auto-generate a short written summary from the analysis above."))

cells.append(code("""top_team = win_counts.idxmax()
toss_win_pct = toss_impact.get(True, 0)
best_venue = venue_runs.idxmax()
top_scorer = top_batters.iloc[0]
top_wicket_taker = top_bowlers.iloc[0]

report = f\"\"\"
T20 WORLD CUP ANALYTICS — SUMMARY REPORT
=========================================
Matches analyzed: {len(matches)}
Most successful team: {top_team} ({win_counts.max()} wins)
Toss winner also won the match in {toss_win_pct:.1f}% of matches
Highest-scoring venue (avg. total match runs): {best_venue} ({venue_runs.max():.0f} runs)

Top run scorer: {top_scorer['player']} ({top_scorer['team']}) — {top_scorer['runs']} runs, SR {top_scorer['strike_rate']}
Top wicket taker: {top_wicket_taker['player']} ({top_wicket_taker['team']}) — {top_wicket_taker['wickets']} wickets, Econ {top_wicket_taker['economy_rate']}

MODEL PERFORMANCE
------------------
Best model: {best_name}
Accuracy: {best['accuracy']:.1%}
ROC-AUC: {best['auc']:.3f}
Most predictive feature: {importances.index[0]}
\"\"\"
print(report)

with open("../t20_wc_summary_report.txt", "w") as f:
    f.write(report)
print("Saved to t20_wc_summary_report.txt")
"""))

cells.append(md("""## 7. Next Steps / Ideas to Extend

- Swap in a **real** T20 World Cup dataset (Kaggle / ESPNcricinfo scrape) using the same column schema.
- Add ball-by-ball data for richer features (powerplay runs, death-over economy, partnership data).
- Try gradient boosting (XGBoost/LightGBM) and hyperparameter tuning.
- Build a Best-XI selector using batting/bowling summary tables.
- Turn this into a Streamlit/Dash interactive dashboard.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("/home/claude/t20wc_project/notebook/T20_WorldCup_Analytics.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook written.")
