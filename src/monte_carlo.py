import gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

 
def print_values(V, size=8):
    i = 0
    
    print("\n\t\t State Value")

    for _ in range(size):
        
        print("------------------------------------------------")

        for _ in range(size):
            
            v = V[i]
            i += 1
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
            
        print("")
        
    print("------------------------------------------------")


def print_policy(policy, size=8):
    
    actions = ['l', 's', 'r', 'n']
    
    i = 0
    
    print("\n\t\t Policy/Actions")

    for _ in range(size):
        print("------------------------------------------------")

        for _ in range(size):
            a = actions[policy[i]]
            i += 1
            print("  %s  |" % a, end="")
            
        print("")
        
    print("------------------------------------------------")
    

def print_stats(env, policy):

    wins = 0
    r = 100
    for i in range(r):
        
        w = _run(env, policy)[-1][-1]
        
        if w == 1:
            wins += 1
    
    print(wins/r)
    

def _argmax(array):
    
    # Finding the action with maximum value                
    indices = [i for i, x in enumerate(array) if x == max(array)]

    return random.choice(indices)
                

def _sel_action(action, all_actions, eps=0.1):
    
    p = np.random.random()
    
    if p < (1 - eps):
        return action
    else:
        return np.random.choice(all_actions)


def _run(env, policy):
    
     env.reset()
     episode = []

     while True:
        
        init_state = env.env.s

        # Select the action
        action = _sel_action(policy[init_state], list(range(env.action_space.n-1)))
                
        # Run the action
        state, reward, done, _ = env.step(action)
        
        # Add step to the episode
        episode.append([init_state, action, reward])
        
        if done:
            break
          
     return episode


def _learn_mc(env, episodes=100, gamma=0.99):

    # Create an arbitrary policy
    policy = [random.randint(0, env.action_space.n-1) for _ in range(env.observation_space.n)]
    
    # Initialize state-action
    Q = [[0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)]
    
    # Create the returns based on state-action pairs
    returns = {(s, a):[] for s in range(env.observation_space.n) for a in range(env.action_space.n)}
    
    deltas = []

    for t in tqdm(range(episodes)):
        
        G = 0
        biggest_change = 0
        
        # Run an episode
        episode = _run(env, policy)
                
        for i in reversed(range(len(episode))): 
            
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            
            # Cummulative discounted rewards
            G = gamma*G + r_t
            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]:

                # Add return to the state/action
                returns[state_action].append(G)
                
                old_q = Q[s_t][a_t]
            
                # Average reward across episodes
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) 
                
                biggest_change = max(biggest_change, abs(old_q - Q[s_t][a_t]))
            
                # Update the policy
                policy[s_t] = _argmax(Q[s_t])
                        
        deltas.append(biggest_change)
        
    V = []
    
    for q in Q:
        V.append(max(q))

    return policy, V, (deltas)


def main():
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    
    # Learn a policy with MC
    policy, V, stats = _learn_mc(env, episodes=20000)

    env.render()
        
    print_values(V)
    print_policy(policy)
    print_stats(env, policy)
    
    # Plot the learning
    plt.plot(stats)
    plt.show()
    

if __name__ == '__main__':
    
   main()
   
   
