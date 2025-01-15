import pandas as pd
import streamlit as st
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Data Loading
# ----------------------------
def load_data(file_name="player_data.csv"):
    """
    Load player data from a CSV file and return a list of dictionaries.
    """
    return pd.read_csv(file_name).to_dict("records")


# ----------------------------
# Optimization Function with Uniqueness Constraints
# ----------------------------
def optimize_lineup(players, min_salary, max_salary, 
                    prev_lineup_sets=None, min_unique=None, 
                    extra_exclusions=[]):
    """
    Solve the lineup optimization problem with salary constraints and,
    if provided, uniqueness constraints relative to previously generated lineups.
    
    Args:
        players: list of player dictionaries.
        min_salary: minimum total salary allowed.
        max_salary: maximum total salary allowed.
        prev_lineup_sets: list of sets, where each set contains the names of players 
                          from a previously generated lineup.
        min_unique: minimum number of different players compared to any previous lineup.
        extra_exclusions: list of players to exclude outright.
        
    Returns:
        A list of player dictionaries representing the lineup; or None if no 
        optimal solution is found.
    """
    # Initialize the optimization problem
    problem = LpProblem("Tennis_Lineup_Optimization", LpMaximize)
    
    # Create a decision variable for each player (1 if selected, 0 otherwise)
    player_vars = {player["Player"]: LpVariable(player["Player"], cat="Binary") for player in players}
    
    # Objective: maximize total projected points
    problem += lpSum(player_vars[player["Player"]] * player["Pts"] for player in players), "Total_Points"
    
    # Salary constraints
    problem += lpSum(player_vars[player["Player"]] * player["Salary"] for player in players) >= min_salary, "Min_Salary"
    problem += lpSum(player_vars[player["Player"]] * player["Salary"] for player in players) <= max_salary, "Max_Salary"
    
    # Exactly 6 players must be selected
    problem += lpSum(player_vars[player["Player"]] for player in players) == 6, "Lineup_Size"
    
    # Exclude additional players as necessary
    for excl in extra_exclusions:
        if excl in player_vars:
            problem += player_vars[excl] == 0, f"Exclude_{excl}"
    
    # Uniqueness constraints: for each previous lineup, force the new lineup to have at least min_unique players different.
    if prev_lineup_sets and min_unique is not None:
        for idx, prev_set in enumerate(prev_lineup_sets):
            relevant_vars = [player_vars[p] for p in prev_set if p in player_vars]
            problem += lpSum(relevant_vars) <= 6 - min_unique, f"Unique_constraint_{idx}"
    
    # Match constraints: Prevent both a player and their opponent from being in the same lineup
    match_dict = {player["Player"]: player["Opponent"] for player in players}
    for player, opponent in match_dict.items():
        if player in player_vars and opponent in player_vars:
            problem += player_vars[player] + player_vars[opponent] <= 1, f"Match_constraint_{player}_{opponent}"
    
    # Solve the optimization problem
    problem.solve(PULP_CBC_CMD(msg=0))
    
    if LpStatus[problem.status] != "Optimal":
        return None
    return [player for player in players if player_vars[player["Player"]].value() == 1]

# ----------------------------
# Lineup Generation Function
# ----------------------------
def generate_lineups(players, min_salary, max_salary, num_lineups, min_unique):
    """
    Generate a number of lineups meeting salary constraints and ensuring that each new lineup
    is sufficiently unique relative to every previously generated lineup.
    
    Args:
        players: list of player dictionaries.
        min_salary: minimum total salary.
        max_salary: maximum total salary.
        num_lineups: number of lineups to generate.
        min_unique: minimum number of unique players compared to any previous lineup.
    
    Returns:
        List of valid lineups (each lineup is a list of player dictionaries).
    """
    lineups = []
    prev_lineup_sets = []  # We'll store frozensets of player names for each lineup
    
    for i in range(num_lineups):
        candidate = optimize_lineup(
            players,
            min_salary,
            max_salary,
            prev_lineup_sets=prev_lineup_sets,
            min_unique=min_unique,
            extra_exclusions=[]
        )
        if candidate is None:
            # If we cannot generate a candidate given the constraints, break out.
            break
        lineups.append(candidate)
        candidate_set = frozenset(player["Player"] for player in candidate)
        prev_lineup_sets.append(candidate_set)
    return lineups


# ----------------------------
# Metrics and Exposure Calculation
# ----------------------------
def calculate_metrics(lineups):
    """
    Compute metrics (total salary and total projected points) for the top, bottom, and median lineups.
    
    Returns:
        Three tuples: (Total Salary, Total Projected Points) for top, bottom, and median lineups.
    """
    lineup_points = [sum(player["Pts"] for player in lineup) for lineup in lineups]
    lineup_points_sorted = sorted(lineup_points)
    
    top_lineup = max(lineups, key=lambda lineup: sum(player["Pts"] for player in lineup))
    bottom_lineup = min(lineups, key=lambda lineup: sum(player["Pts"] for player in lineup))
    median_val = lineup_points_sorted[len(lineup_points_sorted) // 2]
    median_lineup = next(lineup for lineup in lineups if sum(player["Pts"] for player in lineup) == median_val)
    
    top_salary = sum(player["Salary"] for player in top_lineup)
    top_points = sum(player["Pts"] for player in top_lineup)
    bottom_salary = sum(player["Salary"] for player in bottom_lineup)
    bottom_points = sum(player["Pts"] for player in bottom_lineup)
    median_salary = sum(player["Salary"] for player in median_lineup)
    median_points = sum(player["Pts"] for player in median_lineup)
    
    return (top_salary, top_points), (bottom_salary, bottom_points), (median_salary, median_points)


def calculate_exposures(lineups, total_lineups):
    """
    Calculate each player's exposure (in percent) over all generated lineups.
    """
    player_counts = {}
    for lineup in lineups:
        for player in lineup:
            player_counts[player["Player"]] = player_counts.get(player["Player"], 0) + 1

    exposures = [
        {"Player": player, "Exposure (%)": (count / total_lineups) * 100}
        for player, count in sorted(player_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    return pd.DataFrame(exposures)


def count_unique_players(lineups):
    """
    Count the total unique players used across all lineups.
    """
    unique_players = {player["Player"] for lineup in lineups for player in lineup}
    return len(unique_players)


# ----------------------------
# Visualization Functions
# ----------------------------
def plot_lineup_scatter(lineups):
    """
    Plot a scatter graph of lineup number versus total projected points.
    """
    lineup_numbers = list(range(1, len(lineups) + 1))
    lineup_points = [sum(player["Pts"] for player in lineup) for lineup in lineups]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(lineup_numbers, lineup_points, alpha=0.7, c='blue')
    plt.title("Lineup Projected Points by Lineup Number")
    plt.xlabel("Lineup Number")
    plt.ylabel("Total Projected Points")
    plt.grid(True)
    st.pyplot(plt)


# ----------------------------
# Streamlit App Main Function
# ----------------------------
def main():
    st.set_page_config(page_title="Tennis Lineup Optimizer", layout="wide")
    st.title("Tennis Lineup Optimizer with Uniqueness Constraints")
    
    # Sidebar settings
    st.sidebar.header("Settings")
    players = load_data()  # Ensure player_data.csv is available
    
    num_lineups = st.sidebar.slider("Number of Lineups", min_value=1, max_value=50, value=20)
    min_salary = st.sidebar.slider("Minimum Salary Cap", min_value=40000, max_value=50000, value=49000, step=500)
    max_salary = st.sidebar.slider("Maximum Salary Cap", min_value=40000, max_value=50000, value=50000, step=500)
    min_unique = st.sidebar.slider("Minimum Unique Players per Lineup", min_value=1, max_value=6, value=3,
                                   help=("Each new lineup will have at least this many players different than any previous lineup. "
                                         "For example, with 3 unique players, any two lineups can share at most 3 players."))
    
    if st.sidebar.button("Generate Lineups"):
        lineups = generate_lineups(players, min_salary, max_salary, num_lineups, min_unique)
        
        if lineups:
            st.success(f"Generated {len(lineups)} unique lineups!")
            
            # Display metrics
            top, bottom, median = calculate_metrics(lineups)
            st.write("### Lineup Metrics")
            st.write(f"**Top Projected Lineup:** Salary = ${top[0]}, Points = {top[1]:.2f}")
            st.write(f"**Bottom Projected Lineup:** Salary = ${bottom[0]}, Points = {bottom[1]:.2f}")
            st.write(f"**Median Projected Lineup:** Salary = ${median[0]}, Points = {median[1]:.2f}")
            
            # Unique players count
            unique_players_count = count_unique_players(lineups)
            st.write(f"### Total Unique Players Used: {unique_players_count}")
            
            # Scatterplot
            st.write("### Lineup Strength Scatterplot")
            plot_lineup_scatter(lineups)
            
            # Player exposures
            exposures = calculate_exposures(lineups, len(lineups))
            st.write("### Player Exposures")
            st.dataframe(exposures)
        else:
            st.error("Could not generate valid lineups under the current constraints. "
                     "Try lowering the uniqueness requirement or adjusting salary constraints.")

            
if __name__ == "__main__":
    main()
