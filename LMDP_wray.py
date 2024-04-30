# import the LMDP structure from csv files

# lexicographic mdp class.

# filtering actions

# lexocographic value iteration

import numpy as np

class LexicographicMDP:
    def __init__(self, num_states, num_actions, transition_probabilities, rewards, discount_factor, slack_variables, convergence_threshold):
        self.num_states = num_states
        self.num_actions = num_actions
        self.T = transition_probabilities  # Transition matrix [state, action, state']
        self.R = rewards  # Reward functions for each objective [objective, state, action, state']
        self.gamma = discount_factor
        self.slack_variables = slack_variables
        self.epsilon = convergence_threshold
        self.num_objectives = len(rewards)
        self.V = np.zeros((self.num_objectives, num_states))  # Value functions for each objective

    def bellman_update(self, state, objective):
        Q_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for next_state in range(self.num_states):
                transition_prob = self.T[state, action, next_state]
                reward = self.R[objective][state, action, next_state]
                Q_values[action] += transition_prob * (reward + self.gamma * self.V[objective, next_state])

        best_action_value = np.max(Q_values)
        eta = (1 - self.gamma) * self.slack_variables[objective]
        viable_actions = Q_values >= best_action_value - eta
        return np.max(Q_values[viable_actions])

    def value_iteration(self):
        stable = False
        while not stable:
            stable = True
            new_V = np.copy(self.V)
            for objective in range(self.num_objectives):
                for state in range(self.num_states):
                    updated_value = self.bellman_update(state, objective)
                    if np.abs(updated_value - new_V[objective, state]) > self.epsilon:
                        stable = False
                    new_V[objective, state] = updated_value
            self.V = new_V
        return self.V

    def solve(self):
        values = self.value_iteration()
        return values

# Example usage (parameters must be adapted for specific problem instances)
num_states = 10
num_actions = 3
num_objectives = 2

# Randomly generated example data
np.random.seed(42)
transition_probabilities = np.random.rand(num_states, num_actions, num_states)
transition_probabilities /= transition_probabilities.sum(axis=2, keepdims=True)  # Normalize to make probabilities

rewards = np.random.rand(num_objectives, num_states, num_actions, num_states)
discount_factor = 0.95
slack_variables = [0.05, 0.1]  # Slack for each objective
convergence_threshold = 0.01

lmdp = LexicographicMDP(num_states, num_actions, transition_probabilities, rewards, discount_factor, slack_variables, convergence_threshold)
optimal_values = lmdp.solve()
print("Optimal values for each objective:\n", optimal_values)
