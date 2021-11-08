import gym
import numpy as np
import os
import time
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


def print_state_values_approximator(env, Q, size=8):
    
    x, w = Q
    
    print("\n\t\t State Value")
    
    s = 0
    for _ in range(size):
        
        print("------------------------------------------------")

        for _ in range(size):
            
            # Get the max value for state
            q = np.asarray([np.dot(w[a], x(s)) for a in range(env.action_space.n)])
            
            v = np.max(q)
            
            if v >= 0:
                print(" %.4f|" % v, end="")
            else:
                print("%.4f|" % v, end="")
                
            s += 1
            
        print("")
        
    print("------------------------------------------------")


def print_policy_approximator(env, Q, size=8):
    
    x, w = Q
    
    actions_names = ['l', 's', 'r', 'n']
    
    s = 0
    
    print("\n\t\t Policy/Actions")

    for _ in range(size):
        print("------------------------------------------------")

        for _ in range(size):
            
            # Get the max value for state
            q = np.asarray([np.dot(w[a], x(s)) for a in range(env.action_space.n)])
                   
            # Get the best action
            best_action = np.argmax(q)

            s += 1
            print("  %s  |" % actions_names[best_action], end="")
            
        print("")
        
    print("------------------------------------------------")
    

def generate_stats(env, Q):
    
    x, w = Q
        
    wins = 0
    r = 100
    for i in range(r):
           
        e, reward_array = _run(env, x, w, eps=0)
        
        #print(e)
           
        if reward_array[-1] == 1:
            wins += 1
    
    return wins/r
    
    
def play(env, Q):
    
    x, w = Q
    
    _run(env, x, w, eps=0, display=True)
    
    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.plot(data)
    plt.show()
                    

def _run(env, x, w, eps_params=None, display=False):
    
     env.reset()
     episode = []
     reward_array = []
     count = 0
     
     eps=0
     
     if display:
        env.render()
     
     while True:
        
        state = env.env.s
            
        # Select the action prob
        p = np.random.random()
        
        # If epsilon greedy params is defined
        if eps_params is not None:
            
            n0, n = eps_params
            
            # Define the epsilon
            eps = n0/(n0 + n[state])
        
        # epsilon-greedy for exploration vs exploitation
        if p < (1 - eps):
                        
            q = np.asarray([np.dot(w[a], x(state)) for a in range(env.action_space.n)])
                                    
            action = np.random.choice(np.argwhere(q == np.max(q)).ravel())
            
        else:
            action = np.random.choice(env.action_space.n)
       
            
        # Run the action
        _, reward, done, _ = env.step(action)
        
        # Add step and reward
        episode.append((state, action))
        reward_array.append(reward)
        
        count += 1
        
        if display:
            #os.system('clear')
            env.render()
            time.sleep(1)
    
        if done or count > 100:
            break
          
     return episode, reward_array, eps


def _learn_mc_approximator(env, episodes, gamma, n0, disable_tqdm=True):

    featurizer = FeatureUnion([
       ("rbf2", RBFSampler(gamma=1, n_components=10)),
       ("rbf3", RBFSampler(gamma=0.5, n_components=10))
     	])
    
    scaler = StandardScaler()
    
    X = []
    for i in range(3000):
        r = np.random.randint(0, 64)
        
        X.append([r, r%8])

    # Collect observations
    X = np.asarray(X)
    
    # Fit the feature function vector
    scaler.fit(X)
    featurizer.fit(scaler.transform(X))
    
    # Generate the feature funtion
    x = lambda state: featurizer.transform(scaler.transform(np.asarray([[state, state%8]])))[0]
                        
    # Get the feature vector shape
    m = x(0).shape
    
    # Initialize weight vector
    w = np.zeros((4, m[0]))
    
    # Number of visits for each state
    n = {s:0 for s in range(env.observation_space.n)}
    
    # Number of action's selections for state
    na = {(s, a):0 for s in range(env.observation_space.n) for a in range(env.action_space.n)}
                    
    stats = {'return':[], 'cumGTrain':0}

    tic = time.time()
    
    with tqdm(total=episodes, disable=disable_tqdm) as pbar:
        
        G = 0
        sumG = 0
        
        for t in range(episodes):
                   
            episode, rewards, eps = _run(env, x, w, eps_params=(n0, n))
            
            for i, state_action in enumerate(episode):

                state, action = state_action

                # Increment the state visits
                n[state] = n[state] + 1
                
                # Increment the action selection
                na[state_action] = na[state_action] + 1
                
                # Compute the alpha
                alpha = 5/na[state_action]

                x_s = x(state)
                
                G = np.dot(np.array(rewards[i:]), np.fromfunction(lambda i: gamma ** i, (len(rewards) - i , )))
                sumG += G
                
                w[action] += alpha* (G - np.dot(w[action], x_s)) * x_s
                      
            stats['return'].append(G)
        
            toc = time.time()
                    
            pbar.set_description(("{:.1f}s - sumG: {:.6f}, alpha: {:.6f}, gamma: {:.6f}, eps: {:.6f}".format((toc-tic), sumG, alpha, gamma, eps)))
                    
            # Update the bar
            pbar.update(1)
            
        stats['cumGTrain'] = sumG

    return (x, w), stats


def train_approximator(stochastic, episodes, gamma, n0):
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    # Reset the seed
    np.random.seed(2)
    env.seed(2)
    
    # Learn a policy with MC
    Q, stats = _learn_mc_approximator(env, episodes=episodes, gamma=gamma, n0=n0, disable_tqdm=False)
    
    # Plot stats
    plot_episode_return(stats['return'])
    
    return Q, env
    

def grid_search_approximator(stochastic):
    
    if stochastic:
        param_grid = {'alpha': [0.1, 0.01, 0.001], 'gamma': [1, 0.9, 0.1], 'eps':[0.9, 0.5, 0.1], 'episodes': [1000]}
    else:
        param_grid = {'alpha': [0.1, 0.01, 0.001], 'gamma': [1, 0.9, 0.1], 'eps':[0.9, 0.5, 0.1], 'episodes': [1000]}
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    results = pandas.DataFrame(columns=['alpha', 'gamma', 'eps', 'episodes', 'Total G Train', 'win/loss (%)', 'elapsed time (s)'])
    
    for c in ParameterGrid(param_grid):
        
        # Reset the seed
        np.random.seed(412)
        env.seed(412)

        tic = time.time()

        # Learn policy
        Q, stats = _learn_mc_approximator(env, **c, disable_tqdm=True)

        toc = time.time()
        
        elapsed_time = toc - tic
        
        # Generate wins
        win = generate_stats(env, Q)*100
        
        new_row = {'alpha': c['alpha'],
                   'gamma': c['gamma'],
                   'eps': c['eps'],
                   'episodes': c['episodes'],
                   'Total G Train': stats['cumGTrain'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)
        
    print(results)

if __name__ == '__main__':
    
    #grid_search_approximator(stochastic=True)
    #exit()
    
    Q, env = train_approximator(stochastic=False, episodes=3000, gamma=1, n0=100)
    
    print_state_values_approximator(env, Q)
    
    print_policy_approximator(env, Q)
    
    print(generate_stats(env, Q)*100)
   
    play(env, Q)
   
   
