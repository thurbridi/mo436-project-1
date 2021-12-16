import gym
import numpy as np
import os
import random
import time
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
np.random.seed(42)
random.seed(42)

 
def print_state_values_tabular(Q, size=8):

    # Get the max value for state
    V = [max(q_s) for q_s in Q]
    
    print("\n\t\t State Value")
    
    s = 0
    for _ in range(size):
        
        print("------------------------------------------------")

        for _ in range(size):
            
            if V[s] >= 0:
                print(" %.2f|" % V[s], end="")
            else:
                print("%.2f|" % V[s], end="")
                
            s += 1
            
        print("")
        
    print("------------------------------------------------")


def print_policy_tabular(Q, size=8):
    
    actions_names = ['l', 's', 'r', 'n']
    
    i = 0
    
    print("\n\t\t Policy/Actions")

    for _ in range(size):
        print("------------------------------------------------")

        for _ in range(size):
            
            # Get the best action
            best_action = _argmax(Q[i])

            i += 1
            print("  %s  |" % actions_names[best_action], end="")
            
        print("")
        
    print("------------------------------------------------")
    

def generate_stats(env, Q):

    wins = 0
    r = 100
    for i in range(r):
        
        w = _run(env, Q)[-1][-1]
        
        if w == 1:
            wins += 1
    
    return wins/r
    

def play(env, Q):
    
    _run(env, Q, display=True)
    
    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.plot(data)
    plt.show()
    
    
def _argmax(Q):
    
    # Find the action with maximum value                
    actions = [a for a, v in enumerate(Q) if v == max(Q)]

    return random.choice(actions)
                

def _run(env, Q, eps_params=None, display=False):
    
     env.reset()
     episode = []
     
     eps = 0
     
     if display:
        env.render()

     while True:
        
        state = env.env.s
        
        # If epsilon greedy params is defined
        if eps_params is not None:
            
            n0, n = eps_params
            
            # Define the epsilon
            eps = n0/(n0 + n[state])
    
        # Select the action prob
        p = np.random.random()
        
        # epsilon-greedy for exploration vs exploitation
        if p < (1 - eps):
            action = _argmax(Q[state])
        else:
            action = np.random.choice(env.action_space.n)
                
        # Run the action
        _, reward, done, _ = env.step(action)
        
        # Add step to the episode
        episode.append([state, action, reward])
        
        if display:
            os.system('clear')
            env.render()
            time.sleep(1)
        
        if done:
            break
          
     return episode


def _learn_mc_tabular(env, episodes, gamma, n0, disable_tqdm=False):
    
    # Initialize state-action
    Q = [[0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)]
        
    # Number of visits for each state
    n = {s:0 for s in range(env.observation_space.n)}
    
    # Number of action's selections for state
    na = {(s, a):0 for s in range(env.observation_space.n) for a in range(env.action_space.n)}
    
    stats = {'return':[]}

    for t in tqdm(range(episodes), disable=disable_tqdm):
        
        G = 0
        
        # Run an episode
        episode = _run(env, Q, eps_params=(n0, n))
                
        for i in reversed(range(len(episode))): 
            
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            
            # Cummulative discounted rewards
            G = gamma*G + r_t
            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:
                
                # Increment the state visits
                n[s_t] = n[s_t] + 1
                
                # Increment the action selection
                na[state_action] = na[state_action] + 1
            
                # Compute the alpha
                alpha = 1/na[state_action]
            
                # Update the action-value
                Q[s_t][a_t] = Q[s_t][a_t] + alpha*(G - Q[s_t][a_t])
                
            if i == len(episode)-1:
                stats['return'].append(G)

    return Q, stats


def train_tabular(stochastic, episodes=10000, gamma=0.9, n0=10):
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    # Reset the seed
    np.random.seed(42)
    random.seed(42)
    env.seed(42)
    
    # Learn a policy with MC
    Q, stats = _learn_mc_tabular(env, episodes=episodes, gamma=gamma, n0=n0, disable_tqdm=False)
        
    # Plot stats
    plot_episode_return(stats['return'])
    
    return Q, env
    

def grid_search_tabular(stochastic):
    
    if stochastic:
        param_grid = {'n0': [1, 100, 1000, 10000], 'gamma': [1, 0.9, 0.1], 'episodes': [10000, 100000]}
    else:
        param_grid = {'n0': [0.1, 1, 10], 'gamma': [1, 0.9, 0.5, 0.1], 'episodes': [100, 1000]}
            
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=['n0', 'gamma', 'episodes', 'win/loss (%)', 'elapsed time (s)'])
        
    for c in ParameterGrid(param_grid):
        
        # Reset the seed
        np.random.seed(42)
        random.seed(42)
        env.seed(42)

        tic = time.time()

        # Learn policy
        Q, stats = _learn_mc_tabular(env, **c, disable_tqdm=True)
                
        toc = time.time()
        
        elapsed_time = toc - tic
        
        # Generate wins
        win = generate_stats(env, Q)*100
        
        new_row = {'n0': c['n0'],
                   'gamma': c['gamma'],
                   'episodes': c['episodes'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)
        
    print(results)

if __name__ == '__main__':

    Q, env = train_tabular(stochastic=False, episodes=100, gamma=0.9, n0=1)
    
    play(env, Q)
    
    #print_state_values_tabular(Q)
    
    #print(generate_stats(env, Q)*100)
   
