import gym
import numpy as np
import time
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
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
            q = np.asarray([np.dot(w, x((s, a))) for a in range(env.action_space.n)])
            
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
            q = np.asarray([np.dot(w, x((s, a))) for a in range(env.action_space.n)])
                   
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
           
        e, reward_array = _run(env, x, w, eps=0.1)
        
        #print(e)
           
        if reward_array[-1] == 1:
            wins += 1
    
    return wins/r
    
    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.plot(data)
    plt.show()
                    

def _run(env, x, w, eps):
    
     env.reset()
     episode = []
     reward_array = []
     
     while True:
        
        state = env.env.s
            
        # Select the action prob
        p = np.random.random()
        
        # epsilon-greedy for exploration vs exploitation
        if p < (1 - eps):
                        
            q = np.asarray([np.dot(w, x((state, a))) for a in range(env.action_space.n)])
                        
            action = np.random.choice(np.argwhere(q == np.max(q)).ravel())
            
        else:
            action = np.random.choice(env.action_space.n)
       
            
        # Run the action
        _, reward, done, _ = env.step(action)
        
        # Add step and reward
        episode.append((state, action))
        reward_array.append(reward)
                
        if done:
            break
          
     return episode, reward_array


def _learn_mc_approximator(env, episodes, gamma, alpha, disable_tqdm=False):

    featurizer = RBFSampler(gamma=1, random_state=1)
    scaler = StandardScaler()

    # Collect observations
    X = np.asarray([[np.random.randint(0, 64), np.random.randint(0, 4)] for x in range(30000)])
    
    # Fit the feature function vector
    scaler.fit(X)
    featurizer.fit(scaler.transform(X))
    
    # Generate the feature funtion
    x = lambda state_action: featurizer.transform(scaler.transform(np.asarray([state_action])))[0]
                        
    # Get the feature vector shape
    m = x((0, 0)).shape
    
    # Initialize weight vector
    w = np.zeros(m[0])
                
    stats = {'return':[], 'cumGTrain':0}

    tic = time.time()
    
    with tqdm(total=episodes, disable=disable_tqdm) as pbar:
        
        G = 0
        sumG = 0
        
        for t in range(episodes):
            
            eps = episodes / (episodes + 5*t)
            #alpha = episodes//10 / (episodes + 10*t)
                   
            episode, rewards = _run(env, x, w, eps=eps)
            
            for i, state_action in enumerate(episode):

                x_s = x(state_action)
                
                G = np.dot(np.array(rewards[i:]), np.fromfunction(lambda i: gamma ** i, (len(rewards) - i , )))
                sumG += G
                
                w_delta = (G - np.dot(w, x_s)) * x_s
                w_delta = np.minimum(np.maximum(w_delta, -1e+5), 1e+5)
                
                w += alpha*w_delta
                      
            stats['return'].append(G)
        
            toc = time.time()
                    
            pbar.set_description(("{:.1f}s - sumG: {:.6f}, alpha: {:.6f}, gamma: {:.6f}, eps: {:.6f}".format((toc-tic), sumG, alpha, gamma, eps)))
                    
            # Update the bar
            pbar.update(1)
            
        stats['cumGTrain'] = sumG
    
    return (x, w), stats


def train_approximator(stochastic, episodes=10000, gamma=1, alpha=0.001):
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    # Reset the seed
    np.random.seed(2)
    env.seed(2)
    
    # Learn a policy with MC
    Q, stats = _learn_mc_approximator(env, episodes=episodes, gamma=gamma, alpha=alpha, disable_tqdm=True)
    
    # Plot stats
    plot_episode_return(stats['return'])
    
    return Q, env
    

def grid_search_approximator(stochastic):
    
    if stochastic:
        param_grid = {'alpha': [0.1, 0.001, 0.0005], 'gamma': [1, 0.9, 0.1], 'episodes': [1000, 10000]}
    else:
        param_grid = {'alpha': [0.1, 0.001, 0.0005], 'gamma': [1, 0.9, 0.1], 'episodes': [1000, 10000]}
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    results = pandas.DataFrame(columns=['alpha', 'gamma', 'episodes', 'Total G Train', 'win/loss (%)', 'elapsed time (s)'])
    
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
                   'episodes': c['episodes'],
                   'Total G Train': stats['cumGTrain'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)
        
    print(results)

if __name__ == '__main__':
    
    grid_search_approximator(stochastic=False)
    #train_approximator(stochastic=False)
   
   
