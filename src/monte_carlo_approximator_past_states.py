import gym
import numpy as np
import time
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
    

def generate_stats(env, Q):
    
    x, w = Q
        
    wins = 0
    r = 100
    for i in range(r):
           
        _, ws = _run(env, Q, eps=0)
                   
        if ws[-1] == 1:
            wins += 1

    return wins/r
    

def play(env, Q):
    
    _run(env, Q, eps=0, display=True)
    
    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.plot(data)
    plt.show()
    

def _run(env, Q, eps, display=False):
    
     x, w = Q
    
     env.reset()
     steps = []
     reward_array = []
     prev_state = 0
     count = 0 
          
     if display:
        env.render()

     while True:
        
        state = env.env.s
            
        # Select the action prob
        p = np.random.random()
                
        # epsilon-greedy for exploration vs exploitation
        if p < (1 - eps):
            
            q = np.asarray([np.dot(w[a], x((prev_state, state))) for a in range(env.action_space.n)])

            action = np.random.choice(np.argwhere(q == np.max(q)).ravel())
            #print(q, action)
                                         
        else:
            action = np.random.choice(env.action_space.n)
                
        # Run the action
        _, reward, done, _ = env.step(action)
        
        # Add step to the episode
        steps.append([(prev_state, state), action])
        reward_array.append(reward)
        
        prev_state = state
        
        count += 1
        
        if display:
            env.render()
            time.sleep(1)
        
        if done:
            break
        
          
     return steps, reward_array
    
    
def _learn_mc_approximator(env, episodes, gamma, alpha, eps, disable_tqdm=False):
    
    def x(state):
        
        prev_state, next_state = state
        
        prev_state = (prev_state+1)/16
        next_state = (next_state+1)/16
        mod_state = ((next_state+1)%4)/16
        left_state = (16-(next_state+1))/16
        
        features = np.array([1, prev_state, next_state, prev_state*next_state, prev_state**2, next_state**2])
                  
        #norm = features / (np.linalg.norm(features))
  
        #print(norm)
        
        return features
                        
    # Get the feature vector shape
    m = x((0, 0)).shape
    
    # Initialize weight vector
    w = np.zeros((4, m[0]))
            
    stats = {'return':[]}

    for t in tqdm(range(episodes), disable=disable_tqdm):
        
        # Run an episode
        steps, rewards = _run(env, (x, w), eps=eps)
                
        for i, state_action in enumerate(steps):
            
            state, action = state_action
                        
            x_s = x(state)
                
            G = np.dot(np.array(rewards[i:]), np.fromfunction(lambda i: gamma ** i, (len(rewards) - i , )))
                        
            w[action] = w[action] + alpha*(G - np.dot(w[action], x_s)) * x_s
                
            if i == len(steps)-1:
                stats['return'].append(G)
    print(w)
    return (x, w), stats


def train_approximator(stochastic, episodes, gamma, alpha, eps):
    
    env = gym.make('FrozenLake-v1', is_slippery=stochastic)
    
    # Reset the seed
    np.random.seed(2)
    env.seed(2)
    
    # Learn a policy with MC
    Q, stats = _learn_mc_approximator(env, episodes=episodes, gamma=gamma, alpha=alpha, eps=eps, disable_tqdm=False)
        
    # Plot stats
    plot_episode_return(stats['return'])
    
    return Q, env


def grid_search_approximator(stochastic):
    
    if stochastic:
        param_grid = {'alpha': [0.1, 0.001, 0.0005], 'gamma': [1, 0.9, 0.1], 'episodes': [1000, 10000]}
    else:
        param_grid = {'alpha': [0.01, 0.001, 0.0005], 'gamma': [1, 0.9, 0.1], 'eps': [0.9, 0.5, 0.1], 'episodes': [20000]}
    
    env = gym.make('FrozenLake-v1', is_slippery=stochastic)
    
    results = pandas.DataFrame(columns=['alpha', 'gamma', 'eps', 'episodes', 'win/loss (%)', 'elapsed time (s)'])
    
    for c in ParameterGrid(param_grid):
        
        # Reset the seed
        np.random.seed(2)
        env.seed(2)

        tic = time.time()

        # Learn policy
        Q, stats = _learn_mc_approximator(env, **c, disable_tqdm=False)

        toc = time.time()
        
        elapsed_time = toc - tic
        
        # Generate wins
        win = generate_stats(env, Q)*100
        
        new_row = {'alpha': c['alpha'],
                   'gamma': c['gamma'],
                   'eps': c['eps'],
                   'episodes': c['episodes'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)
        
    print(results)


if __name__ == '__main__':
    

    grid_search_approximator(False)
    exit()
    
    Q, env = train_approximator(stochastic=False, episodes=50000, gamma=1, alpha=0.01, eps=0.3)
    
    play(env, Q)
    
    print(generate_stats(env, Q)*100)
   
