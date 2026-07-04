"""
generate_dataset.py
--------------------
Generates a realistic T20 World Cup dataset for analytics/ML practice.

NOTE ON DATA SOURCE:
Live scraping of Kaggle/GitHub hosted T20 World Cup CSVs was not reliably
accessible from this environment, so this script PROGRAMMATICALLY GENERATES
a dataset that mirrors the real structure of T20 World Cup data (real team
names, real venues, realistic scoring patterns, toss effects, and team
strength differentials modeled on actual ICC T20I rankings tiers). Match
outcomes are simulated, not real results.

If you have (or find) a real dataset -- e.g. Kaggle's T20 World Cup CSVs --
just replace data/matches.csv with your file, keeping the same column names,
and every notebook cell downstream will work unchanged.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Teams with an approximate relative "strength" rating (loosely modeled on
#    real-world T20I tiers as of recent World Cups). Higher = stronger.
# ---------------------------------------------------------------------------
TEAMS = {
    "India": 95, "England": 90, "Australia": 89, "South Africa": 87,
    "New Zealand": 85, "Pakistan": 84, "West Indies": 78, "Sri Lanka": 75,
    "Afghanistan": 76, "Bangladesh": 70, "Ireland": 62, "Scotland": 58,
    "Netherlands": 60, "Zimbabwe": 57, "Namibia": 52, "USA": 50,
    "Nepal": 48, "UAE": 46, "Canada": 42, "Papua New Guinea": 40,
    "Oman": 41, "Uganda": 38,
}

VENUES = [
    ("Eden Gardens", "Kolkata", "India"), ("Melbourne Cricket Ground", "Melbourne", "Australia"),
    ("Kensington Oval", "Bridgetown", "Barbados"), ("Newlands", "Cape Town", "South Africa"),
    ("Dubai International Stadium", "Dubai", "UAE"), ("Nassau County International Stadium", "New York", "USA"),
    ("Arnos Vale Ground", "Kingstown", "St Vincent"), ("Providence Stadium", "Georgetown", "Guyana"),
    ("Sydney Cricket Ground", "Sydney", "Australia"), ("Wankhede Stadium", "Mumbai", "India"),
    ("Old Trafford", "Manchester", "England"), ("R Premadasa Stadium", "Colombo", "Sri Lanka"),
]

PLAYER_POOL = {  # a handful of representative players per team
    "India": ["R. Sharma", "V. Kohli", "S. Yadav", "H. Pandya", "J. Bumrah", "R. Jadeja", "A. Kishan"],
    "England": ["J. Buttler", "J. Bairstow", "B. Stokes", "L. Livingstone", "A. Rashid", "M. Wood"],
    "Australia": ["D. Warner", "M. Marsh", "G. Maxwell", "P. Cummins", "A. Zampa", "M. Starc"],
    "South Africa": ["Q. de Kock", "A. Markram", "D. Miller", "K. Rabada", "A. Nortje", "T. Shamsi"],
    "New Zealand": ["K. Williamson", "D. Conway", "G. Phillips", "T. Boult", "T. Southee", "M. Santner"],
    "Pakistan": ["B. Azam", "M. Rizwan", "S. Afridi", "F. Zaman", "H. Ali", "S. Khan"],
    "West Indies": ["N. Pooran", "S. Hope", "R. Powell", "A. Russell", "A. Joseph", "S. Hetmyer"],
    "Sri Lanka": ["P. Nissanka", "K. Mendis", "W. Hasaranga", "D. Chameera", "C. Asalanka"],
    "Afghanistan": ["R. Gurbaz", "R. Zadran", "M. Nabi", "R. Khan", "N. Zadran"],
    "Bangladesh": ["S. Al Hasan", "L. Das", "N. Hossain", "M. Mustafizur", "T. Iqbal"],
    "Ireland": ["P. Stirling", "H. Tector", "M. Adair", "B. White"],
    "Scotland": ["G. Munsey", "M. Cross", "B. Wheal", "M. Leask"],
    "Netherlands": ["M. O'Dowd", "S. Edwards", "B. Klaassen", "P. van Meekeren"],
    "Zimbabwe": ["S. Williams", "R. Burl", "B. Muzarabani", "W. Madhevere"],
    "Namibia": ["G. Erasmus", "D. Wiese", "J. Frylinck"],
    "USA": ["M. Patel", "A. Gous", "S. Netravalkar", "N. Kapoor"],
    "Nepal": ["R. Paudel", "K. Bhurtel", "S. Lamichhane"],
    "UAE": ["C. Ashraf", "V. Vagh", "M. Waseem"],
    "Canada": ["N. Dutta", "S. Baldwin", "K. Ramgoolam"],
    "Papua New Guinea": ["A. Vala", "S. Kariko", "N. Vanua"],
    "Oman": ["Z. Khan", "A. Kaleem", "K. Naveed"],
    "Uganda": ["B. Musinguzi", "F. Achelam", "R. Sekandi"],
}

STAGES = ["Group Stage"] * 30 + ["Super 8"] * 12 + ["Semi Final"] * 2 + ["Final"] * 1
N_MATCHES = len(STAGES)

start_date = datetime(2026, 2, 10)


def simulate_score(strength, overs=20, opp_strength=70):
    """Simulate a first/second innings T20 total based on team strength."""
    base = 130 + (strength - 70) * 1.1
    base += RNG.normal(0, 14)
    base -= (opp_strength - 70) * 0.35  # stronger bowling opposition suppresses score
    runs = int(max(60, min(240, base)))
    wickets = int(np.clip(RNG.normal(6, 2.2), 0, 10))
    return runs, wickets


def simulate_match(match_id, date, stage, team1, team2):
    s1, s2 = TEAMS[team1], TEAMS[team2]
    venue_name, city, country = VENUES[RNG.integers(0, len(VENUES))]
    toss_winner = RNG.choice([team1, team2])
    toss_decision = RNG.choice(["Bat", "Field"], p=[0.35, 0.65])

    team1_runs, team1_wkts = simulate_score(s1, opp_strength=s2)
    # chasing team gets a small boost/penalty based on required rate pressure
    team2_runs, team2_wkts = simulate_score(s2, opp_strength=s1)

    # Win probability leans on strength differential + small home/toss edge
    strength_diff = (s1 - s2) / 20
    toss_bonus = 0.15 if toss_winner == team1 else -0.15
    p_team1 = 1 / (1 + np.exp(-(strength_diff + toss_bonus)))
    winner = team1 if RNG.random() < p_team1 else team2

    # Adjust the losing team's score down slightly for narrative consistency
    if winner == team1 and team2_runs >= team1_runs:
        team2_runs = max(60, team1_runs - RNG.integers(3, 25))
    elif winner == team2 and team1_runs >= team2_runs:
        team1_runs = max(60, team2_runs - RNG.integers(3, 25)) if RNG.random() < 0.5 else team1_runs
        team2_wkts = min(10, team2_wkts + 1) if team2_runs > team1_runs else team2_wkts

    margin_type = RNG.choice(["runs", "wickets"])
    if margin_type == "runs":
        margin = abs(team1_runs - team2_runs)
        margin_str = f"{margin} runs"
    else:
        margin = RNG.integers(1, 9)
        margin_str = f"{margin} wickets"

    pom_team = winner
    player_of_match = RNG.choice(PLAYER_POOL[pom_team])

    return {
        "match_id": match_id,
        "date": date.strftime("%Y-%m-%d"),
        "stage": stage,
        "venue": venue_name,
        "city": city,
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "team1_score": team1_runs,
        "team1_wickets": team1_wkts,
        "team1_overs": 20.0,
        "team2_score": team2_runs,
        "team2_wickets": team2_wkts,
        "team2_overs": 20.0,
        "winner": winner,
        "win_margin": margin_str,
        "player_of_match": player_of_match,
    }


def build_matches():
    team_names = list(TEAMS.keys())
    matches = []
    used_pairs = set()
    date = start_date
    for i, stage in enumerate(STAGES):
        # pick two teams, avoid exact repeat pairs where possible
        for _ in range(50):
            t1, t2 = RNG.choice(team_names, size=2, replace=False)
            pair = tuple(sorted([t1, t2]))
            if pair not in used_pairs or len(used_pairs) > 150:
                used_pairs.add(pair)
                break
        m = simulate_match(i + 1, date, stage, t1, t2)
        matches.append(m)
        date += timedelta(days=1 if RNG.random() < 0.7 else 2)
    return pd.DataFrame(matches)


def build_player_batting(matches_df):
    """Aggregate a simple batting summary table across the tournament."""
    records = []
    for team, players in PLAYER_POOL.items():
        team_matches = matches_df[(matches_df.team1 == team) | (matches_df.team2 == team)]
        n = len(team_matches)
        if n == 0:
            continue
        for p in players:
            innings = max(1, int(n * RNG.uniform(0.6, 1.0)))
            avg_runs = RNG.uniform(15, 55) * (TEAMS[team] / 80)
            runs = int(np.clip(RNG.normal(avg_runs * innings, avg_runs * 0.3), 20, 600))
            balls = int(runs / RNG.uniform(1.0, 1.6)) + innings * 5
            fours = int(runs * RNG.uniform(0.08, 0.16))
            sixes = int(runs * RNG.uniform(0.03, 0.09))
            fifties = int(np.clip(runs // 55 * RNG.uniform(0.4, 1.0), 0, innings))
            hundreds = 1 if runs > 350 and RNG.random() < 0.15 else 0
            highest = int(np.clip(runs / max(1, innings) * RNG.uniform(1.3, 2.2), 20, 122))
            strike_rate = round(runs / max(1, balls) * 100, 2)
            records.append({
                "player": p, "team": team, "innings": innings, "runs": runs,
                "balls_faced": balls, "fours": fours, "sixes": sixes,
                "fifties": fifties, "hundreds": hundreds, "highest_score": highest,
                "strike_rate": strike_rate,
                "batting_average": round(runs / max(1, innings * RNG.uniform(0.75, 1.0)), 2),
            })
    return pd.DataFrame(records)


def build_player_bowling(matches_df):
    records = []
    for team, players in PLAYER_POOL.items():
        team_matches = matches_df[(matches_df.team1 == team) | (matches_df.team2 == team)]
        n = len(team_matches)
        if n == 0:
            continue
        for p in players:
            innings = max(1, int(n * RNG.uniform(0.5, 0.95)))
            overs = round(innings * RNG.uniform(2.5, 4.0), 1)
            economy = round(RNG.uniform(5.8, 9.5), 2)
            runs_conceded = int(overs * economy)
            wickets = int(np.clip(RNG.normal(innings * 1.1, innings * 0.5), 0, innings * 4))
            best_bowling_w = int(np.clip(RNG.poisson(2), 1, 5))
            best_bowling_r = int(RNG.uniform(8, 35))
            economy_rate = economy
            bowling_avg = round(runs_conceded / max(1, wickets), 2) if wickets else None
            records.append({
                "player": p, "team": team, "innings": innings, "overs": overs,
                "runs_conceded": runs_conceded, "wickets": wickets,
                "economy_rate": economy_rate, "bowling_average": bowling_avg,
                "best_bowling": f"{best_bowling_w}/{best_bowling_r}",
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    matches_df = build_matches()
    batting_df = build_player_batting(matches_df)
    bowling_df = build_player_bowling(matches_df)

    matches_df.to_csv("data/matches.csv", index=False)
    batting_df.to_csv("data/batting_summary.csv", index=False)
    bowling_df.to_csv("data/bowling_summary.csv", index=False)

    print("Matches:", matches_df.shape)
    print("Batting summary:", batting_df.shape)
    print("Bowling summary:", bowling_df.shape)
