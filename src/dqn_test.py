import argparse

import gym
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from tqdm import trange


from dqn.agent import Agent
from model.architecture import Net
from utils import *


def run(env, episodes, steps, gamma, e, modelpkl):
    # Initialize history memory
    step_list = []
    reward_list = []
    loss_list = []
    e_list = []
    best_actions = []

    state_space = env.observation_space.n
    action_space = env.action_space.n

    agent = Agent(e, gamma, state_space, action_space, Net(state_space, action_space))
    agent.load_model(modelpkl)
    # agent.train(True)
    
    for i in trange(episodes):
        state = int(env.reset())
        reward_all = 0
        done = False
        s = 0
        total_loss = 0
        actions = []

        for s in range(steps):
            state = Variable(OH(state, state_space))

            # propose an action
            action = agent.select_action(state)

            # Save actions
            actions.append(action)

            # what are the consequences of taking that action?
            new_state, reward, done, _ = env.step(action)

            # if we're dead
            if done and (reward == 0.0):
                reward = -1

            # move to next state
            reward_all += reward

            # Render
            os.system('clear')
            env.render()
            time.sleep(0.5)

            #Breaks
            if reward == 1:
                best_actions.append(actions)
                break

            if done:
                break

            state = new_state

        # logging epochs
        step_list.append(s)
        reward_list.append(reward_all)

    return step_list, reward_list, best_actions
	

if __name__ == "__main__": 

    # Environment
    env = gym.make('FrozenLake8x8-v0', is_slippery=False)

    # Reset the seed
    np.random.seed(2)
    env.seed(2)
    
    # Create directory to save models

    ## Modelo 5
    episodes = 1
    steps = 200
    gamma = 0.97
    e = 0.01

    #actions
    best_actions = []

    # Predicting
    #for i in range(24):
    #name = str(i)
    modelpkl = "dqn_models_deterministic2/net_params_5000_200_0.97_0.01_0.01.pkl"    # model 5 , deterministic2

    step_list, reward_list, best_actions = run(env,episodes, steps, gamma, e, modelpkl)