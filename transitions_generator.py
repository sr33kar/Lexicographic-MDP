def generate_transition_function(size=4):
    # List to store transitions
    transitions = []

    # Iterate over all possible states in the grid
    for x in range(size):
        for y in range(size):
            # Define all four actions and compute the results
            for action in ['R', 'L', 'U', 'D']:
                unintended_states = []
                current_state = (x, y)
                if action == 'R':
                    count = 1
                    next_state = (x, min(y+1, size-1))
                    transitions.append((current_state, action, next_state, 0.8))
                    # Add intended state transition
                    if current_state != next_state:
                        transitions.append((current_state, action, current_state, 0.1))
                        count +=1
                    unintended_states = [(x, min(y+1, size-1)), (x, max(y-1, 0)), (max(x-1, 0), y), (min(x+1, size-1),y)]
                    # Add unintended state transitions
                    added =[current_state, next_state]
                    for unintended in unintended_states:
                        if unintended != next_state and unintended not in added and count <=3:
                            transitions.append((current_state, action, unintended, 0.1))
                            count +=1
                            added.append(unintended)
                elif action == 'L':
                    count = 1
                    next_state = (x, max(y-1, 0))
                    transitions.append((current_state, action, next_state, 0.8))
                    # Add intended state transition
                    if current_state != next_state:
                        transitions.append((current_state, action, current_state, 0.1))
                        count +=1
                    unintended_states = [(x, min(y+1, size-1)), (x, max(y-1, 0)), (max(x-1, 0), y), (min(x+1, size-1),y)]
                    # Add unintended state transitions
                    added =[current_state, next_state]
                    for unintended in unintended_states:
                        if unintended != next_state and unintended not in added and count <=3:
                            transitions.append((current_state, action, unintended, 0.1))
                            count +=1
                            added.append(unintended)
                elif action == 'U':
                    count = 1
                    next_state = (max(x-1, 0), y)
                    transitions.append((current_state, action, next_state, 0.8))
                    # Add intended state transition
                    if current_state != next_state:
                        transitions.append((current_state, action, current_state, 0.1))
                        count+=1
                    unintended_states = [(x, min(y+1, size-1)), (x, max(y-1, 0)), (max(x-1, 0), y), (min(x+1, size-1),y)]
                    added =[current_state, next_state]
                    for unintended in unintended_states:
                        if unintended != next_state and unintended not in added and count<=3:
                            transitions.append((current_state, action, unintended, 0.1))
                            added.append(unintended)
                elif action == 'D':
                    count = 1
                    next_state = (min(x+1, size-1), y)
                    transitions.append((current_state, action, next_state, 0.8))
                    # Add intended state transition
                    if current_state != next_state:
                        transitions.append((current_state, action, current_state, 0.1))
                        count +=1
                    unintended_states = [(x, min(y+1, size-1)), (x, max(y-1, 0)), (max(x-1, 0), y), (min(x+1, size-1),y)]
                    # Add unintended state transitions
                    added =[current_state, next_state]
                    for unintended in unintended_states:
                        if unintended != next_state and unintended not in added and count<=3:
                            transitions.append((current_state, action, unintended, 0.1))
                            count+=1
                            added.append(unintended)
                
                

    return transitions

# Generate the transition function
transition_function = generate_transition_function()

# Print transitions
for transition in transition_function:
    print('('+str(transition[0][0])+' '+str(transition[0][1])+ ')' + "," + transition[1] + "," +'('+str(transition[2][0])+' '+str(transition[2][1])+ ')' + "," + str(transition[3]))
