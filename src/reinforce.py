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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal


class FrozenLake(nn.Module):

    def __init__(self, env, hidden):

        super().__init__()
        
        self.env = env

        # Create the networks
        self.reinforce = ReinforceNetwork(hidden)
        self.baseliner = BaselineNetwork(hidden)

    def forward(self):
                
        action_array = []
        log_pi_array = []
        baseline_array = []
        reward_array = []
        
        self.env.reset()
        
        previous_state = 0
        action = 0
        
        for t in range(100):
            
            state = torch.tensor([action, self.env.env.s], dtype=torch.float)
                        
            # Generate the next action
            log_pi, action = self.reinforce(state)
            
            action_array.append(action)
            
            # Add the next observation to array
            log_pi_array.append(log_pi)
            
            # Generate the baseline
            #b_t = self.baseliner(state).squeeze()
            
            # Add the baseline to array
            #baseline_array.append(b_t)
                                
            # Run the action
            _, reward, done, _ = self.env.step(int(action))
            
            reward_array.append(torch.tensor(reward, dtype=torch.float))
            
            previous_state = int(state[1])
            action = int(action)
        
            if done:                
                break
   
        # Stack the lists
        #baseline_array = torch.stack(baseline_array)
        action_array = torch.stack(action_array)
        log_pi_array = torch.stack(log_pi_array)
        reward_array = torch.stack(reward_array)
                
        return reward_array, action_array, log_pi_array, baseline_array
    
    def play(self):
        
        self.env.reset()
        self.env.render()
        
        action = 0
    
        while True:
        
            state = torch.tensor([action, self.env.env.s], dtype=torch.float)
                        
            # Generate the next action
            _, action = self.reinforce(state)
                                                
            # Run the action
            _, reward, done, _ = self.env.step(int(action))
            
            action = int(action)
        
            #os.system('clear')
            self.env.render()
            time.sleep(1)
            
            if done:
                break
            
    
    def print_state_values(self, size=8):
        
        print("\n\t\t State Value")
        
        s = 0
        for _ in range(size):
            
            print("------------------------------------------------")

            for _ in range(size):
                            
                v = self.baseliner(torch.tensor([0, s], dtype=torch.float))
                
                if v >= 0:
                    print(" %.4f|" % v, end="")
                else:
                    print("%.4f|" % v, end="")
                    
                s += 1
                
            print("")
            
        print("------------------------------------------------")


    def print_policy(self, size=8):
                
        actions_names = ['l', 's', 'r', 'n']
        
        s = 0
        
        print("\n\t\t Policy/Actions")

        for _ in range(size):
            print("------------------------------------------------")

            for _ in range(size):

                # Get the best action
                _, best_action = self.reinforce(torch.tensor([0, s], dtype=torch.float))

                s += 1
                print("  %s  |" % actions_names[best_action], end="")
            
            print("")
            
        print("------------------------------------------------")
        

    def generate_stats(self):
                
        wins = 0
        r = 100
        for i in range(r):
            
            reward_array, _, _, _ = self.forward()
            
            if reward_array[-1] == 1:
                wins += 1
        
        return wins/r
        
        
class ReinforceNetwork(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.fc_mu_1 = nn.Linear(2, hidden)
        self.fc_mu_2 = nn.Linear(hidden, 4)
        
    def forward(self, state):
               
        mu = torch.relu(self.fc_mu_1(state))
        mu = self.fc_mu_2(mu)
                
        probs = F.softmax(mu)   
        
        action = probs.multinomial(4).data
      
        prob = probs[action[0]]
        
        return prob.log(), action[0]
    

class BaselineNetwork(nn.Module):

    def __init__(self, hidden):
        super().__init__()

        self.fc_1 = nn.Linear(2, hidden)
        self.fc_2 = nn.Linear(hidden, 1)

    def forward(self, state):
        
        b_t = torch.relu(self.fc_1(state))
        b_t = self.fc_2(b_t)
        
        return b_t

    
def plot_episode_return(data):
    
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.plot(data)
    plt.show()
                    

def _learn_reinforce(env, episodes, gamma, alpha, hidden, disable_tqdm=True):

    model = FrozenLake(env, hidden)

    reward_array = []
    loss_reinforce_array = []
    loss_baseline_array = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    loss_mse = torch.nn.MSELoss()
    
    model.train()
    
    sumG = 0

    tic = time.time()
    with tqdm(total=episodes) as pbar:
        
        # For each minibatch
        for i  in range(episodes):
                                    
            # Call the model and pass the minibatch
            rewards, action_array, log_pi_array, baseline_array = model()

            # Convert list to tensors and reshape
            baselines = baseline_array
            log_pi = log_pi_array
                        
            #loss_baseline = loss_mse(baselines, R)
            
            R = torch.zeros(1, 1)
            loss = 0
            
            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                loss = loss - (log_pi[i]*(R)).sum() #- (0.0001*entropies[i]).sum()
                
            loss = loss / len(rewards)
            
            # Compute reinforce loss
            #adjusted_reward = R #- baselines.detach()

            #loss_reinforce = torch.sum(log_pi * R)
            
            # Join the losses
            #loss = -loss_reinforce #+ loss_baseline      
            
            optimizer.zero_grad()
            
            # Update the weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
                            
            optimizer.step()
            
            # Store the losses
            reward_array.append(torch.sum(R).cpu().data.numpy())
            
            sumG += torch.sum(R).cpu().data.numpy()

            # Measure elapsed time
            toc = time.time()

            # Set the var description
            pbar.set_description(("{:.1f}s - train RL: {:.6f} - G: {:.1f}".format((toc-tic), loss.item(), sumG)))
            
            # Update the bar
            pbar.update(1)
            
    return model, reward_array
                

def train_reinforce(stochastic, episodes=10000, gamma=0.9, alpha=0.0001, hidden=128):
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    # Reset the seed
    np.random.seed(2)
    torch.manual_seed(2)
    env.seed(2)
    
    # Learn a policy with REINFORCE
    model, stats = _learn_reinforce(env, episodes=episodes, gamma=gamma, alpha=alpha, hidden=hidden, disable_tqdm=False)
    
    # Plot stats
    plot_episode_return(stats)
    
    return model, env
    

def grid_search_reinforce(stochastic):
    
    if stochastic:
        param_grid = {'alpha': [0.1, 0.001, 0.0001], 'gamma': [1], 'hidden': [32, 64, 128], 'episodes': [1000]}
    else:
        param_grid = {'alpha': [0.0001, 0.00001], 'gamma': [1, 0.9], 'hidden': [64, 128], 'episodes': [5000, 10000]}
    
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)
    
    results = pandas.DataFrame(columns=['alpha', 'gamma', 'hidden', 'episodes', 'total_r', 'win/loss (%)', 'elapsed time (s)'])
    
    for c in ParameterGrid(param_grid):
        
        # Reset the seed
        np.random.seed(2)
        torch.manual_seed(2)
        env.seed(2)

        tic = time.time()

        # Learn policy
        model, stats = _learn_reinforce(env, **c, disable_tqdm=True)

        toc = time.time()
        
        elapsed_time = toc - tic
        
        # Generate wins
        win = model.generate_stats()*100
        
        new_row = {'alpha': c['alpha'],
                   'gamma': c['gamma'],
                   'hidden': c['hidden'],
                   'episodes': c['episodes'],
                   'total_r': sum(stats),
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)
        
    print(results)


if __name__ == '__main__':
    
    grid_search_reinforce(False)
    
    exit()
    
    model, env = train_reinforce(stochastic=False, episodes=5000)
    
    model.print_state_values()
    
    model.print_policy()
    
    print(model.generate_stats()*100)
   
    model.play()

