#!/usr/bin/env python
# coding: utf-8

# In[140]:


import csv

# Function to read rewards or NSEs from a CSV file
def read_csv_file(filename):
    data = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            state_str = row[0].strip('()').split(' ')
            state = tuple(map(int, state_str[0].split() + state_str[1].split()))
            value = float(row[1])
            data[state] = value
    return data

# Read rewards and NSEs into dictionaries
rewards_data = read_csv_file('rewards_1.csv')
nses_data = read_csv_file('nse_1.csv')

# Transition data
transition_data = {}
with open('transitions_1.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        current_state = tuple(map(int, row[0].strip('()').split(' ')))
        action = row[1]
        next_state = tuple(map(int, row[2].strip('()').split(' ')))
        probability = float(row[3])
        if current_state not in transition_data:
            transition_data[current_state] = {}
        if action not in transition_data[current_state]:
            transition_data[current_state][action] = []
        transition_data[current_state][action].append((probability, next_state))


# In[138]:


# LexicographicMDP class
class LexicographicMDP:
    def __init__(self, transitions, rewards, nses, gamma=0.9):
        self.transitions = transitions
        self.rewards = rewards
        self.nses = nses
        self.gamma = gamma

    def R(self, state):
        reward = self.rewards.get(state, 0)
        nse = self.nses.get(state, 0)
        return reward, nse

    def T(self, state, action):
        return self.transitions.get(state, {}).get(action, [])

    def actions(self, state):
        return ['R', 'U', 'D', 'L', 'EXIT']

# Value iteration function
def value_iteration(mdp, epsilon=0.001):
    V = {}
    for state in mdp.rewards.keys():
        V[state] = 0
    while True:
        delta = 0
        for state in mdp.rewards.keys():
            v = V[state]
            reward, nse = mdp.R(state)  # Get reward and NSE for the current state
            # Consider both reward and NSE in value iteration
            V[state] = max(sum(prob * ((reward + nse) + mdp.gamma * V[next_state]) for prob, next_state in mdp.T(state, action)) for action in mdp.actions(state))
            delta = max(delta, abs(v - V[state]))
        if delta < epsilon:
            break
    return V

# Derive policy function
def derive_policy(mdp, V):
    policy = {}
    for state in mdp.rewards.keys():
        policy[state] = max(mdp.actions(state), key=lambda action: sum(prob * ((mdp.R(next_state)[0] + mdp.R(next_state)[1]) + mdp.gamma * V[next_state]) for prob, next_state in mdp.T(state, action)))
    return policy    
    

# Function to estimate slack
def estimate_slack(mdp, V, Omega, epsilon=0.001):
    # Assuming Omega is provided as {(state, action): penalty}
    modified_transitions = {}
    for state in mdp.transitions:
        modified_transitions[state] = {}
        for action in mdp.transitions[state]:
            if (state, action) not in Omega or Omega[(state, action)] == 0:
                modified_transitions[state][action] = mdp.transitions[state][action]
    
    original_transitions = mdp.transitions
    mdp.transitions = modified_transitions

    V_hat = value_iteration(mdp, epsilon)

    mdp.transitions = original_transitions
    
    slack = max(abs(V[state] - V_hat.get(state, 0)) for state in V)
    return slack

# Function to run the LMDP including slack estimation
def run_lmdp():
    mdp = LexicographicMDP(transition_data, rewards_data, nses_data)

    V = value_iteration(mdp)

    # Define Omega based on nses data with a threshold for significant NSE
    Omega = {(state, action): mdp.nses.get(state, 0) for state in mdp.rewards for action in mdp.actions(state) if mdp.nses.get(state, 0) > 0.1}  # Define your own threshold

    slack = estimate_slack(mdp, V, Omega)
    print("Estimated Slack:", slack)

    policy = derive_policy(mdp, V)

    for state, action in policy.items():
        print(f"State {state} - Action {action}")



# In[139]:


# Run the MDP
run_lmdp()


# In[141]:


# Run the MDP
run_lmdp()

