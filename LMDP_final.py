# imports
import numpy as np
import pandas as pd

####################################################################################################################################
# Load data from CSV files
states_df = pd.read_csv('nse.csv')
rewards_primary_df = pd.read_csv('rewards.csv')  # Assume this is the primary objective
rewards_secondary_df = states_df  # Secondary objective from nse.csv
transitions_df = pd.read_csv('transitions.csv', header=None, names=['state', 'action', 'next_state', 'probability'])

# Process states and actions
states = list(states_df['state'].unique())
actions = list(transitions_df['action'].unique())

####################################################################################################################################
# Initialize reward structures
def read_state_values(filename):
    state_values = {}
    df = pd.read_csv(filename)
    for index, row in df.iterrows():
        state = row[0]
        value = float(row[1])
        state_values[state] = value
    return state_values

file1 = 'rewards.csv'
file2 = 'nse.csv'

state_values_1 = read_state_values(file1)
state_values_2 = read_state_values(file2)

rewards = [state_values_1, state_values_2]


########################################################################################################################################
# We create the transitions dictionary from the dataframe
transitions = {}

# Iterate over rows in the dataframe to populate the dictionary
for index, row in transitions_df.iterrows():
    s1 = row['state'].strip()
    a = row['action'].strip()
    s2 = row['next_state'].strip()
    p = row['probability']

    if s1 not in transitions:
        transitions[s1] = {}
    if a not in transitions[s1]:
        transitions[s1][a] = {}
    transitions[s1][a][s2] = p

# Function to get s2 states for given s1 and action
def get_s2_states(s1, action):
    if s1 in transitions and action in transitions[s1]:
        return list(transitions[s1][action].keys())
    return []

# Function to get probability for given s1, action, and s2
def get_probability(s1, action, s2):
    if s1 in transitions and action in transitions[s1] and s2 in transitions[s1][action]:
        return float(transitions[s1][action][s2])
    return 0.0

###################################################################################################################################
# bellman backup
def bellman_update(state, value_function, rewards, gamma, feasible_actions):
    """Performs the Bellman update for a given state."""
    best_value = float('-inf')
    for action in feasible_actions:
        sum_over_states = sum(get_probability(state, action, next_state) *
                              (rewards[state] + gamma * value_function[next_state])
                              for next_state in get_s2_states(state, action))  # Update here to transitions to loop through states +modify rewards
        if sum_over_states > best_value:
            best_value = sum_over_states
    return best_value

###########################################################################################################################################
# filtering actions
def filter_actions(states, actions, q_values, eta):
    """Filters actions for each state based on the Q-values and a small slack variable eta."""
    filtered_actions = {}
    for state in states:
        max_q_value = max(q_values[state, action] for action in actions)
        filtered_actions[state] = [action for action in actions if q_values[state, action] >= max_q_value - eta]
    return filtered_actions

############################################################################################################################################
# lexicographic value iteration
def lexicographic_value_iteration(states, actions, rewards, gamma, eta, epsilon, max_iterations):
    objective_count = len(rewards)
    values = []

    for objective in range(objective_count):
        
        #value funtion initialized
        value_function = {}
        for state in states:
            value_function[state] = 0

        for iteration in range(max_iterations):
            new_value_function = value_function.copy()
            for state in states:
                q_values = {}
                
                for action in actions:
                    q_values[state, action] = sum(get_probability(state, action, next_state)  *
                                                  (rewards[objective][state] +
                                                   gamma * value_function[next_state])
                                                  for next_state in get_s2_states(state, action))
                if objective == 0:

                    new_value_function[state] = bellman_update(state, value_function, rewards[objective], gamma, actions)
                else:
                    new_value_function[state] = bellman_update(state, value_function, rewards[objective], gamma, feasible_actions[state])
            
            max = float('-inf')
            for state in states:
                diff = abs(new_value_function[state] - value_function[state])
                if diff > max: max = diff
            if max < epsilon:
                break
            value_function = new_value_function
        values[objective] = value_function
        if objective == 0: feasible_actions = filter_actions(states, actions, q_values, eta)[state] 
    return values

############################################################################################################################################
# policy generation
def derive_policy(states, actions, rewards, final_values, gamma):
    """Derives the optimal policy from the final value functions."""
    policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        
        # Iterate through all possible actions to find the best one for the current state
        for action in actions:
            current_value = sum(get_probability(state, action, next_state) *
                                (rewards[state] + gamma * final_values[next_state])
                                for next_state in get_s2_states(state, action))  # Update here to transitions to loop through states + modify rewards
            if current_value > best_value:
                best_value = current_value
                best_action = action
        
        policy[state] = best_action
    return policy

gamma = 0.95  # Discount factor
delta = 0.2
eta = (1-gamma)*delta    # Slack variable for filtering actions
epsilon = 0.01  # Convergence threshold
max_iterations = 1000  # Max iterations for value iteration

x = lexicographic_value_iteration(states, actions, rewards, gamma, eta, epsilon, max_iterations)

policy = derive_policy(states, actions, rewards, x[1], gamma)

print(policy)