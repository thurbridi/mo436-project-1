import gym
import numpy as np
import random
import time
import pandas
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

 
def print_values(env, Q, size=8):
    
    x, w = Q
    
    print("\n\t\t State Value")
    
    s = 0
    for _ in range(size):
        
        print("------------------------------------------------")

        for _ in range(size):
            
            # Get the max value for state
            q = np.asarray([np.dot(w, x(s, a)[0]) for a in range(env.action_space.n)])
                    
            v = np.max(q)
            
            if v >= 0:
                print(" %.4f|" % v, end="")
            else:
                print("%.4f|" % v, end="")
                
            s += 1
            
        print("")
        
    print("------------------------------------------------")


def print_policy(env, Q, size=8):
    
    x, w = Q
    
    actions_names = ['l', 's', 'r', 'n']
    
    s = 0
    
    print("\n\t\t Policy/Actions")

    for _ in range(size):
        print("------------------------------------------------")

        for _ in range(size):
            
            # Get the max value for state
            q = np.asarray([np.dot(w, x(s, a)[0]) for a in range(env.action_space.n)])
                   
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
                
        if _run(env, x, w, eps=0)[-1][-1] == 1:
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


def _generate_feature_repr(env):
    
    # Create the scaler and feature represention
    scaler = StandardScaler()
    
    #featurizer = FeatureUnion([
            #("rbf1", RBFSampler(gamma=2.0, n_components=10)),
            #("rbf2", RBFSampler(gamma=0.5, n_components=10))
    #])
    
    featurizer = RBFSampler(gamma=1, n_components=100)
    
    # Generate the observations
    observations = np.asarray([(env.observation_space.sample(), env.action_space.sample()) for x in range(30000)])
        
    scaler.fit(observations)
    featurizer.fit(scaler.transform(observations))
    
    return lambda state_action: featurizer.transform(scaler.transform(np.asarray(state_action)))
                

def _run(env, x, w, eps=0.2):
    
     env.reset()
     episode = []
     reward_array = []
     
     while True:
        
        state = env.env.s
            
        # Select the action prob
        p = np.random.random()
        
        # epsilon-greedy for exploration vs exploitation
        if p < (1 - eps):
                        
            q = np.asarray([np.dot(w, x(state, a)[0]) for a in range(env.action_space.n)])
            
            action = np.argmax(q)
            
        else:
            action = np.random.choice(env.action_space.n)
                            
        # Run the action
        _, reward, done, _ = env.step(action)
        
        # Add step to the episode
        episode.append([state, action, reward])
        reward_array.append(reward)
                
        if done:
            break
          
     return episode, reward_array


def _learn_mc_approximator(env, episodes, gamma, alpha):
        
    # Generate the feature representation
    def x(state, action):
        
        features = np.array([[state+1, action+1]]) #, (state+1)*(action+1), ((state+1)**2)*(action+1), (state+1)*((action+1)**2)
        
        #norm = features/np.linalg.norm(features)
        
        return features
    
    # Get the feature vector shape
    _, m = x(0, 0).shape
    
    # Initialize weight vector
    w = np.random.rand(m)
                
    stats = {'return':[], 'V*':[]}

    tic = time.time()
    
    with tqdm(total=episodes) as pbar:
        
        for t in range(episodes):
            
            eps = (episodes / (episodes + t))
            alpha = (episodes//100 / (episodes + t))
            gamma = 1 #(episodes//10 + 0.95 * t) / (episodes//2 + t)
        
            G = 0
            
            # Run an episode
            #episode = _run(env, x, w, eps=eps)
                    
            #for s_t, a_t, r_t in reversed(episode): 
                                                        
                #Cummulative discounted rewards
                #G = gamma*G + r_t
                    
                #Generate the action-value representation
                #x_s = x(s_t, a_t)[0]
                                   
                #Compute the weight delta
                #w_delta = alpha*(G - np.dot(w, x_s))*x_s                                   
                            
                #Normalize w delta
                #w_delta = np.minimum(np.maximum(w_delta, -1e+5), 1e+5)
                                        
                #Update the action-value weights
                #w = w + w_delta
                
            episode, rewards = _run(env, x, w, eps=eps)
            for i, state in enumerate(episode):
                
                s_t, a_t, _ = state
                
                x_s = x(s_t, a_t)[0]
                
                if i + 1 >= len(rewards):
                    break
                
                G = np.dot(np.array(rewards[i + 1:]), np.fromfunction(lambda i: gamma ** i, (len(rewards) - i - 1, )))
                w_delta = (G - np.dot(w, x_s)) * x_s
                
                w_delta = np.minimum(np.maximum(w_delta, -1e+5), 1e+5)
                
                w += alpha*w_delta
                      
            stats['return'].append(G)
        
            toc = time.time()
                    
            pbar.set_description(("{:.1f}s - alpha: {:.6f}, gamma: {:.6f}, epsilon: {:.6f}".format((toc-tic), alpha, gamma, eps)))
                    
            # Update the bar
            pbar.update(1)
            
    print(w)
       
    return (x, w), stats


def main():
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    
    # Learn a policy with MC
    Q, stats = _learn_mc_approximator(env, episodes=30000, gamma=1, alpha=0.001)

    env.render()
        
    print_values(env, Q)
    print_policy(env, Q)
    print(generate_stats(env, Q))
    
    # Plot stats
    plot_episode_return(stats['return'])
    #plot_V(stats['V*'])
    

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
        Q, stats = _learn_mc_tabular(env, **c)
        
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
   
   
