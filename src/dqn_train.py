import argparse

import gym
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from sklearn.model_selection import ParameterGrid
from tqdm import trange


from dqn.agent import Agent
from model.architecture import Net
from utils import *


def run(env, models_folder, episodes, steps, gamma, e, lr):
    # Initialize history memory
    step_list = []
    reward_list = []
    loss_list = []
    e_list = []

    state_space = env.observation_space.n
    action_space = env.action_space.n

    agent = Agent(e, gamma, state_space, action_space, Net(state_space, action_space))
    agent.train(True)
    loss = nn.MSELoss()
    optimizer = optim.Adam(agent.model.parameters(), lr=lr)

    for i in trange(episodes):
        state = int(env.reset())
        reward_all = 0
        done = False
        s = 0
        total_loss = 0

        for s in range(steps):
            state = Variable(OH(state, state_space))

            # propose an action
            action = agent.select_action(state)

            # what are the consequences of taking that action?
            new_state, reward, done, _ = env.step(action)

            # if we're dead
            if done and (reward == 0.0):
                reward = -1

            # store memories for experience replay
            Q1 = agent.model(Variable(OH(new_state, state_space)))
            targetQ = agent.remember(Q1, action, reward)

            # optimize predicting rewards
            output = agent.model(state)
            train_loss = loss(output, targetQ)
            total_loss += train_loss.data

            agent.model.zero_grad()
            train_loss.backward()
            optimizer.step()

            # move to next state
            reward_all += reward
            state = new_state

            if reward==1:
                break;

            # decrease epsilon after success
            if done:
                if reward > 0:
                    agent.epsilon *= 0.9 + 1E-6  # always explore a bit during training
                break

        # logging epochs
        loss_list.append(total_loss / s)
        step_list.append(s)
        reward_list.append(reward_all)
        e_list.append(agent.epsilon)

    agent.save_model(f'{models_folder}/net_params_'+str(episodes)+'_'+str(steps)+'_'+str(gamma)+'_'+str(e)+'_'+str(lr)+'.pkl')
    return step_list, e_list, reward_list, loss_list
	

if __name__ == "__main__": 

    # Environment
    env = gym.make('FrozenLake8x8-v0', is_slippery = False)

    # Reset the seed
    np.random.seed(2)
    env.seed(2)
    
    # Create directory to save models
    models_folder = 'dqn_models_deterministic2'
    make_directory(models_folder)

    #param_grid = {'episodes': [3000, 5000], 'steps': [100], 'gamma': [0.85, 0.9, 0.95, 0.99], 'e':[0.001, 0.01, 0.02, 0.03], 'lr':[0.1, 0.01, 0.001, 0.0001] }
    param_grid = {'episodes': [5000], 'steps': [200], 'gamma': [0.95, 0.97, 0.99], 'e':[0.01, 0.001], 'lr':[0.1, 0.01, 0.001, 0.0001] }
    #param_grid = {'episodes': [50], 'steps': [100], 'gamma': [0.85], 'e':[0.01], 'lr':[0.01] }
    
    results = pd.DataFrame(columns=['comb', 'episodes', 'steps', 'gamma', 'e', 'lr', 'wins', 'Success Ep', 'elapsed time (s)'])

    comb = 0

# GridSearch ParamGrid
    for c in ParameterGrid(param_grid):
        
        # Learning
        tic = time.time()
        step_list, e_list, reward_list, loss_list = run(env, models_folder, **c)
        toc = time.time()
        
        elapsed_time = toc - tic        
        
        wins = sum(i for i in reward_list if i > 0) 
        wins_rate = wins/ c['episodes']

        loss_avg = sum(loss_list) / c['episodes']

        new_row = {'comb': comb,
                   'episodes': c['episodes'],
                   'steps': c['steps'],
                   'gamma': c['gamma'],
                   'e': c['e'],
                   'lr': c['lr'],
                   'wins': wins,
                   'Success Ep': wins_rate,
                   'elapsed time (s)': elapsed_time} 
        
        results = results.append(new_row, ignore_index=True)

        print(new_row)

        # Create directory to save plots
        plots_folder = 'dqn_results_deterministic2'
        make_directory(plots_folder)

        # Plot by window rolling
        plot_results(False, True, plots_folder, c['episodes'], comb, Steps=step_list, Rewards=reward_list, Loss=loss_list, Epsilon=e_list)

        # Plot list of values
        plot_result_lists(False, True, plots_folder, comb, Steps=step_list, Rewards=reward_list, Loss=loss_list, Epsilon=e_list)        
        
        # Number of parameters combination
        comb += 1