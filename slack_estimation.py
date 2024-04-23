import numpy as np

# Define the state space, action space, and the transition function
states = np.array([...])  # Define your states here
actions = np.array([...])  # Define your actions here
transition_function = np.zeros((len(states), len(actions), len(states)))  # Define transitions

# Define rewards and penalties
primary_reward = {...}  # Define the primary reward function
penalty_for_nse = {...}  # Define the penalty for negative side effects

# Define a function to compute the state-value function V*
def compute_state_value_function(states, actions, transition_function, rewards, gamma=0.95):
    V = np.zeros(len(states))
    delta = float('inf')
    
    while delta > 1e-6:
        delta = 0
        for s in range(len(states)):
            v = V[s]
            V[s] = max([sum([transition_function[s][a][s_prime] * (rewards.get((s, a), 0) + gamma * V[s_prime])
                            for s_prime in range(len(states))]) for a in range(len(actions))])
            delta = max(delta, abs(v - V[s]))
    
    return V

# Compute the optimal value function with NSE
V_star_with_nse = compute_state_value_function(states, actions, transition_function, primary_reward)

# Modify the transition function to exclude actions leading to NSE
transition_function_nse_free = np.copy(transition_function)
for s in range(len(states)):
    for a in range(len(actions)):
        if penalty_for_nse.get((s, a), 0) > 0:
            transition_function_nse_free[s][a] = np.zeros(len(states))

# Compute the optimal value function without NSE
V_star_without_nse = compute_state_value_function(states, actions, transition_function_nse_free, primary_reward)

# Calculate slack
slack = np.abs(V_star_with_nse - V_star_without_nse)

print("Calculated slack for each state:", slack)
