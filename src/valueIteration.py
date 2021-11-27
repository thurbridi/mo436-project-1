import numpy as np
import gym
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt



env = gym.make('FrozenLake8x8-v0')
count_actions = env.action_space.n #Actions: Left, Down, Right, Up
count_states = env.observation_space.n #64 states

#Transition Probability
# Each element of the list ([(p, s', r, g)])
# p: Probability of transitioning into the state
# s': next state
# r: Reward
# g: Is the game terminated?
print(env.P[0])

def value_iteration_policy(env, max_iterations=1000, discount_factor=0.9, count_states=64,
							count_actions=4):
	# Init empty action-state value list
    stateValue = [0 for i in range(count_states)]
	# Init empty policy list
    policy = [0 for i in range(count_states)]
    newStateValue = stateValue.copy()

    for i in range(max_iterations):
        for state in range(count_states):
			# Empty vector for all action-state values from the current state
            action_values = []

			# Go through avalivable actions from the current state
            for action in range(count_actions):
				# Init cumulutive variable for the action-state values
                state_value = 0

				# Go through Transition Probability in the chosen step
                for i in range(len(env.P[state][action])):
					# Get Transition Probability values
                    prob, next_state, reward, done = env.P[state][action][i]
					# Action-state value formula
                    state_action_value = prob * (reward + discount_factor*stateValue[next_state])
					# Add action-state value by the value from the current action
                    state_value += state_action_value

				 # Append the value of each action
                action_values.append(state_value)

			# Choose the action which gives the maximum value
            best_action = np.argmax(np.asarray(action_values))
			# Update the value of the state
            newStateValue[state] = action_values[best_action]
			 # Update policy list with the best action
            policy[state] = best_action

		# Overwrite action-state value list
        stateValue = newStateValue.copy()
    return policy


def run_multiple_games(env, policy, episodes=1000):
	#  Init var for counting successes
	successes = 0
	# Init list for counting steps in each scenario
	steps_list = []
	for episode in range(episodes):
		# Generate new game/enviroment. It returns 0 state
		observation = env.reset()
		steps=0
		while True:
			# Take trained action from policy list
			action = policy[observation]
			# Perform action in enviroment
			observation, reward, done, _ = env.step(action)
			steps+=1
			# Game ended with sucess
			if done and reward == 1:
				successes += 1
				steps_list.append(steps)
				break
			elif done and reward == 0:
				#Game over
				break
	return episodes, successes, steps_list

labmda_values = np.arange(88,100,1)
successes_list = []
steps_list = []

for i in range(len(labmda_values)):
    policy = value_iteration_policy(env, max_iterations=4000,
                                    discount_factor=labmda_values[i]/100,
                                    count_states=count_states,
                                    count_actions=count_actions)

    episodes, successes, steps = run_multiple_games(env, policy, episodes=100)
    successes_list.append(successes)
    steps_list.append(steps)

plt.plot(labmda_values/100, successes_list)
plt.title("Successes per discount factor")
plt.xlabel("Discount factor")
plt.ylabel("Successes")
plt.xticks(labmda_values/100)
plt.show()

"""
if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v1')
    obs = env.reset()

    for t in range(1000):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            env.render()
            print(f'Episode finished after {t+1} timesteps')
            break

    env.close()
"""
