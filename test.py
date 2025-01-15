import pandas as pd
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import numpy as np

def load_data():
    """
    Load player data from a CSV file in the same directory.
    """
    file_name = "player_data.csv"  # Hardcoded filename
    return pd.read_csv(file_name).to_dict("records")

def optimize_lineup(players, lineup_size, min_salary, max_salary, excluded_players=[]):
    """
    Optimize a single lineup while excluding specific players and respecting salary constraints.
    """
    problem = LpProblem("Tennis_Lineup_Optimization", LpMaximize)

    # Decision variables for each player (1 if selected, 0 otherwise)
    player_vars = {player["Player"]: LpVariable(player["Player"], cat="Binary") for player in players}

    # Objective: Maximize total projected points
    problem += lpSum(player_vars[player["Player"]] * player["Pts"] for player in players), "Total_Points"

    # Constraints
    # Total salary within min and max
    problem += lpSum(player_vars[player["Player"]] * player["Salary"] for player in players) >= min_salary, "Min_Salary"
    problem += lpSum(player_vars[player["Player"]] * player["Salary"] for player in players) <= max_salary, "Max_Salary"

    # Exactly `lineup_size` players must be selected
    problem += lpSum(player_vars[player["Player"]] for player in players) == lineup_size, "Lineup_Size"

    # Exclude specified players
    for excluded_player in excluded_players:
        problem += player_vars[excluded_player] == 0

    # Solve the optimization problem
    problem.solve()

    if LpStatus[problem.status] != "Optimal":
        return None

    # Extract the optimal lineup
    return [player for player in players if player_vars[player["Player"]].value() == 1]

def generate_hybrid_lineups(players, lineup_size, min_salary, max_salary, num_lineups, unique_constraint):
    """
    Generate lineups using a hybrid of Weighted and Randomized exclusions with looser constraints.
    """
    lineups = []
    player_usage = {player["Player"]: 0 for player in players}
    max_retries = 5  # Maximum number of retries for loosening constraints

    for i in range(num_lineups):
        excluded_players = []

        # Hybrid Exclusion Logic:
        # 1. Exclude players used in 3-4 consecutive lineups for the next 1-2 lineups
        for player, usage in player_usage.items():
            if usage >= 3:
                excluded_players.append(player)

        # 2. Randomly exclude additional players to enforce uniqueness
        if len(lineups) > 0:
            all_players_in_lineups = {p["Player"] for lineup in lineups for p in lineup}
            excluded_players += list(
                np.random.choice(list(all_players_in_lineups), size=min(unique_constraint, len(all_players_in_lineups)), replace=False)
            )

        # Generate a new lineup with retries for loosening constraints
        retries = 0
        lineup = None
        while lineup is None and retries < max_retries:
            lineup = optimize_lineup(players, lineup_size, min_salary, max_salary, excluded_players)
            if lineup is None:
                retries += 1
                unique_constraint = max(0, unique_constraint - 1)  # Loosen uniqueness constraint

        if lineup is None:  # Stop if no valid lineup can be generated
            break

        lineups.append(lineup)
        for player in lineup:
            player_usage[player["Player"]] += 1

    return lineups

def calculate_metrics(lineups):
    """
    Calculate the top, bottom, and median projected lineups.
    """
    lineup_points = [sum(player["Pts"] for player in lineup) for lineup in lineups]
    lineup_points.sort()

    # Top projected lineup
    top_lineup = max(lineups, key=lambda lineup: sum(player["Pts"] for player in lineup))
    top_salary = sum(player["Salary"] for player in top_lineup)
    top_points = sum(player["Pts"] for player in top_lineup)

    # Bottom projected lineup
    bottom_lineup = min(lineups, key=lambda lineup: sum(player["Pts"] for player in lineup))
    bottom_salary = sum(player["Salary"] for player in bottom_lineup)
    bottom_points = sum(player["Pts"] for player in bottom_lineup)

    # Median projected lineup
    median_index = len(lineup_points) // 2
    median_lineup = next(lineup for lineup in lineups if sum(player["Pts"] for player in lineup) == lineup_points[median_index])
    median_salary = sum(player["Salary"] for player in median_lineup)
    median_points = sum(player["Pts"] for player in median_lineup)

    return (top_salary, top_points), (bottom_salary, bottom_points), (median_salary, median_points)

def display_metrics(top, bottom, median):
    """
    Display the salary used and total lineup projection for top, bottom, and median lineups.
    """
    st.write("### Lineup Metrics:")
    st.write(f"**Top Projected Lineup**: Salary = ${top[0]}, Points = {top[1]:.2f}")
    st.write(f"**Bottom Projected Lineup**: Salary = ${bottom[0]}, Points = {bottom[1]:.2f}")
    st.write(f"**Median Projected Lineup**: Salary = ${median[0]}, Points = {median[1]:.2f}")

# Streamlit app
st.title("Hybrid Tennis Lineup Generator")

# Load data from the local file
players = load_data()

# User inputs
lineup_size = st.number_input("Number of Players per Lineup", value=6, step=1, min_value=1)
num_lineups = st.number_input("Number of Lineups to Generate", value=20, step=1, min_value=1)
unique_constraint = st.number_input("Number of Unique Players Between Lineups", value=3, step=1, min_value=0)
min_salary = st.number_input("Minimum Salary Cap", value=49000, step=500)
max_salary = st.number_input("Maximum Salary Cap", value=50000, step=500)

if st.button("Generate Lineups"):
    lineups = generate_hybrid_lineups(players, lineup_size, min_salary, max_salary, num_lineups, unique_constraint)

    if lineups:
        st.success(f"Generated {len(lineups)} lineups using the Hybrid Exclusion Strategy!")
        top, bottom, median = calculate_metrics(lineups)
        display_metrics(top, bottom, median)
    else:
        st.error("Could not generate any valid lineups. Adjust your constraints and try again.")

