from sklearn.model_selection import ParameterGrid
from datetime import datetime 
import numpy as np
import pandas as pd
import gym

env = gym.make("FrozenLake-v0", is_slippery=False, map_name='8x8')
n_observations = env.observation_space.n
n_actions = env.action_space.n

#Initialize the Q-table to 0
Q_table = np.zeros((n_observations,n_actions))
#print(Q_table)

#number of episode we will run
n_episodes = 50000

#maximum of iteration per episode
max_iter_episode = 1000

#discounted factor
gamma = 0.95

#learning rate
lr = 0.001

#initialize the exploration probability to 1
epsilon = 1 #exploration_proba

#exploartion decreasing decay for exponential decreasing
exploration_decreasing_decay = 0.001

# minimum of exploration proba
min_exploration_proba = 0.01

# Initialize list of rewards
rewards = np.zeros(n_episodes)

def grid_search_tabular(comb, lr, gamma, epsilon):
    # #sum the rewards that the agent gets from the environment
    rewards_per_episode = np.zeros(n_episodes)

    for e in range(n_episodes):
        #we initialize the first state of the episode
        current_state = env.reset()
        done = False

        for i in range(max_iter_episode): 
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled float is less than the exploration proba
            #     the agent selects arandom action
            # else
            #     he exploits his knowledge using the bellman equation 
            
            if np.random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q_table[current_state,:])
            
            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)
            
            # We update our Q-table using the Q-learning iteration
            Q_table[current_state, action] = (1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]))
                    
            rewards_per_episode[e] += reward

            # If the episode is finished, we leave the for loop
            if done:
                break
            current_state = next_state

        
        #print("Done, iters, total_episode_reward:", done, aux, total_episode_reward) 

        #We update the exploration proba using exponential decay formula 
        epsilon = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
        
        # Add rewards for this episode
        #if done==True and rewards_per_episode[e]==0.0:
            #rewards_per_episode[e] = -1

    return rewards_per_episode
    
param_grid = {'lr': [0.01, 0.001, 0.0001], 'gamma': [0.8, 0.9, 1.0], 'epsilon':[1.0, 0.7, 0.5]}
table = pd.DataFrame(columns=['c', 'learning_rate', 'gamma', 'epsilon', 'wins', 'elapsed time (s)'])

comb = 0
rewards_per_model = []
np.random.seed(2)
env.seed(2)

for c in ParameterGrid(param_grid):
    print("Combination ", comb)

    t0 = datetime.now()
    rewards = grid_search_tabular(comb, **c)
    elapsed_time = datetime.now() - t0
    
    rewards_per_model.append(rewards)
    wins = (rewards > 0).sum()

    new_row = { 'c' : comb,
                'learning_rate': c['lr'],
                'gamma': c['gamma'],
                'epsilon': c['epsilon'],
                'wins': wins,
                'elapsed time (s)': elapsed_time}

    table = table.append(new_row, ignore_index=True) 
    comb += 1
    print(new_row) 


print(table)