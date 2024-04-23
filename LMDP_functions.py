import numpy as np

def bellman_update(state, value_function, transition_probabilities, rewards, gamma, feasible_actions):
    """Performs the Bellman update for a given state."""
    best_value = float('-inf')
    for action in feasible_actions:
        sum_over_states = sum(transition_probabilities[state][action][next_state] *
                              (rewards[state][action][next_state] + gamma * value_function[next_state])
                              for next_state in range(len(value_function)))  # Update here to transitions to loop through states +modify rewards
        if sum_over_states > best_value:
            best_value = sum_over_states
    return best_value

def filter_actions(states, actions, q_values, eta):
    """Filters actions for each state based on the Q-values and a small slack variable eta."""
    filtered_actions = {}
    for state in states:
        max_q_value = max(q_values[state, action] for action in actions)
        filtered_actions[state] = [action for action in actions if q_values[state, action] >= max_q_value - eta]
    return filtered_actions

def lexicographic_value_iteration(states, actions, transition_probabilities, rewards, gamma, eta, epsilon, max_iterations=1000):
    """Performs lexicographic value iteration for multi-objective MDPs."""
    objective_count = len(rewards)
    values = np.zeros((objective_count, len(states)))

    for objective in range(objective_count):
        value_function = np.zeros(len(states))
        for iteration in range(max_iterations):
            new_value_function = np.copy(value_function)
            for state in states:
                q_values = {}
                for action in actions:
                    q_values[state, action] = sum(transition_probabilities[state][action][next_state] *
                                                   (rewards[objective][state][action][next_state] +
                                                    gamma * value_function[next_state])
                                                   for next_state in states) # Update here to transitions to loop through states + modifiy rewards

                if objective == 0:
                    feasible_actions = actions  # All actions are considered for the first objective
                else:
                    feasible_actions = filter_actions(states, actions, q_values, eta)

                new_value_function[state] = bellman_update(state, value_function, transition_probabilities, rewards[objective], gamma, feasible_actions[state])

            if np.max(np.abs(new_value_function - value_function)) < epsilon:
                break
            value_function = new_value_function
        values[objective] = value_function

    return values

#  function for deriving policy 
def derive_policy(states, actions, transition_probabilities, rewards, final_values, gamma):
    """Derives the optimal policy from the final value functions."""
    policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        
        # Iterate through all possible actions to find the best one for the current state
        for action in actions:
            current_value = sum(transition_probabilities[state][action][next_state] *
                                (rewards[state][action][next_state] + gamma * final_values[next_state])
                                for next_state in states)  # Update here to transitions to loop through states + modify rewards
            if current_value > best_value:
                best_value = current_value
                best_action = action
        
        policy[state] = best_action
    return policy

