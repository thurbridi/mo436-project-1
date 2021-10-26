import gym
import numpy as np
import random
import time
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt

 
def print_values(Q, size=8):

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


def print_policy(Q, size=8):
    
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
    
    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cummulative Reward")
    plt.plot(data)
    plt.show()
    
    
def plot_V(data):
    
    plt.xlabel("Episode")
    plt.ylabel("V*")
    plt.plot(data)
    plt.show()
    
    
def _argmax(Q):
    
    # Find the action with maximum value                
    actions = [a for a, v in enumerate(Q) if v == max(Q)]

    return random.choice(actions)
                

def _run(env, Q, eps_params=None):
    
     env.reset()
     episode = []
     
     eps = 0

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
        
        if done:
            break
          
     return episode


def _learn_mc(env, episodes, gamma, n0):
    
    # Initialize state-action
    Q = [[0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)]
        
    # Number of visits for each state
    n = {s:0 for s in range(env.observation_space.n)}
    
    # Number of action's selections for state
    na = {(s, a):0 for s in range(env.observation_space.n) for a in range(env.action_space.n)}
    
    stats = {'return':[], 'V*':[]}

    for t in tqdm(range(episodes)):
        
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
                
                        
        stats['return'].append(G)
        stats['V*'].append(np.amax(Q))

    return Q, stats


def main():
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    
    # Learn a policy with MC
    Q, stats = _learn_mc(env, episodes=10000, gamma=0.9, n0=10)

    env.render()
        
    print_values(Q)
    print_policy(Q)
    print(generate_stats(env, Q))
    
    # Plot stats
    plot_episode_return(stats['return'])
    plot_V(stats['V*'])
    

def report():
    
    conf = [{'n0': 0.1, 'gamma': 0.9, 'episodes': 1000},
            {'n0': 0.1, 'gamma': 0.9, 'episodes': 10000},
            {'n0': 0.1, 'gamma': 0.5, 'episodes': 1000},
            {'n0': 0.1, 'gamma': 0.5, 'episodes': 10000},
            {'n0': 0.1, 'gamma': 0.1, 'episodes': 1000},
            {'n0': 0.1, 'gamma': 0.1, 'episodes': 10000},
            {'n0': 1, 'gamma': 0.9, 'episodes': 1000},
            {'n0': 1, 'gamma': 0.9, 'episodes': 10000},
            {'n0': 1, 'gamma': 0.5, 'episodes': 1000},
            {'n0': 1, 'gamma': 0.5, 'episodes': 10000},
            {'n0': 1, 'gamma': 0.1, 'episodes': 1000},
            {'n0': 1, 'gamma': 0.1, 'episodes': 10000},
            {'n0': 10, 'gamma': 0.9, 'episodes': 1000},
            {'n0': 10, 'gamma': 0.9, 'episodes': 10000},
            {'n0': 10, 'gamma': 0.5, 'episodes': 1000},
            {'n0': 10, 'gamma': 0.5, 'episodes': 10000},
            {'n0': 10, 'gamma': 0.1, 'episodes': 1000},
            {'n0': 10, 'gamma': 0.1, 'episodes': 10000},
            {'n0': 100, 'gamma': 0.9, 'episodes': 1000},
            {'n0': 100, 'gamma': 0.9, 'episodes': 10000},
            {'n0': 100, 'gamma': 0.5, 'episodes': 1000},
            {'n0': 100, 'gamma': 0.5, 'episodes': 10000},
            {'n0': 100, 'gamma': 0.1, 'episodes': 1000},
            {'n0': 100, 'gamma': 0.1, 'episodes': 10000}]
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    
    results = pandas.DataFrame(columns=['n0', 'gamma', 'episodes', 'win/loss (%)', 'elapsed time (s)'])
    
    Q_array = []
    stats_array = []
    
    for c in conf:

        tic = time.time()

        # Learn policy
        Q, stats = _learn_mc(env, **c)
        
        Q_array.append(Q)
        stats_array.append(stats)
        
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
    
    #report()

    main()
   
   
