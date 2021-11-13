import gym
import numpy as np
import sys
import os
import time
import pandas
import random
import pickle
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


#from utils import plotting
# Deterministic environment FrozenLake-v0
def print_state_values_q(Q, size=8):
    print("\n\t\t\t State Value")

    print("--------------------------------------------------------")
    for i in range(size):
        for j in range(size):
            k = i*size + j
            q = np.max(Q[k])
            if q >= 0:
                print("  %.2f|" % np.max(Q[k]), end="")
            else:
                print(" %.2f|" % np.max(Q[k]), end="")
        print("\n--------------------------------------------------------")

def print_policy_q(Q, size=8):

    actions_names = ['\u2190', '\u2193', '\u2192', '\u2191']

    print("\n\t\t Policy/Actions")

    print("------------------------------------------------")
    for i in range(size):
        for j in range(size):
            k = i*size + j
            q = np.argmax(Q[k])
            print("  %s  |" % actions_names[q], end="")
        print("\n------------------------------------------------")


#Initialize the Q-table to 0
def qlearning_tabular(env, episodes=10000, epsilon_max=1.0, epsilon_decay=0.001, min_epsilon=0.01, gamma_=0.99, alpha=0.1 ):
    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    Q_table = np.zeros((n_observations,n_actions))
    #number of episode we will run
    n_episodes = episodes

    #maximum of iteration per episode
    max_iter_episode = 100

    #initialize the exploration probability to 1
    exploration_proba = epsilon_max

    #exploartion decreasing decay for exponential decreasing
    exploration_decreasing_decay = epsilon_decay

    # minimum of exploration proba
    min_exploration_proba = min_epsilon

    #discounted factor
    gamma = gamma_

    #learning rate
    lr = alpha

    # Initialize list of rewards
    rewards_per_episode = list()

    #we iterate over episodes
    for e in range(n_episodes):
        #we initialize the first state of the episode
        current_state = env.reset()
        done = False

        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0

        for t in itertools.count():
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled flaot is less than the exploration proba
            #     the agent selects arandom action
            # else
            #     he exploits his knowledge using the bellman equation

            if np.random.uniform(0,1) < exploration_proba:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(Q_table[current_state,:])

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)

            if done and reward == 0:
                reward = -1.0

            # We update our Q-table using the Q-learning iteration
            Q_table[current_state, action] = Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:]) - Q_table[current_state, action])
            total_episode_reward = total_episode_reward + reward

            # If the episode is finished, we leave the for loop
            if done:
                break

            current_state = next_state

        #We update the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))

        rewards_per_episode.append(total_episode_reward)

    return Q_table, rewards_per_episode

def qlearning_tabular_test(env, Q, episodes=100, display=False):
    Q_table = Q
    #number of episode we will run
    n_episodes = episodes

    #maximum of iteration per episode
    wins_ = 0

    #we iterate over episodes
    for e in range(n_episodes):
        #we initialize the first state of the episode
        current_state = env.reset()
        done = False

        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0

        for t in itertools.count():
            if display==True:
                env.render()

            action = np.argmax(Q_table[current_state,:])

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)

            # If the episode is finished, we leave the for loop
            if done:
                wins_ += reward
                if display == True:
                    env.render()

                break

            current_state = next_state


    return wins_/n_episodes

def report_sarsa(stochastic):
    if stochastic:
        param_ = {'episodes': [1000, 5000], 'epsilon_decay':[0.01, 0.001, 0.0001], 'alpha':[0.001, 0.01, 0.1], 'gamma':[1.0, 0.99, 0.9, 0.0]}
    else:
        param_ = {'episodes': [1000, 5000], 'epsilon_decay':[0.01, 0.001, 0.0001], 'alpha': [0.001, 0.01, 0.1], 'gamma': [1.0, 0.99, 0.9, 0.0]}

    #Environment
    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=['episodes', 'epsilon_decay', 'alpha', 'gamma', 'win/loss (%)', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        #print(c)
        # Reset the seed
        np.random.seed(42)
        random.seed(42)
        env.seed(42)

        tic = time.time()

        # Learn policy
        Q, stats = qlearning_tabular(env, episodes=c['episodes'], epsilon_decay=c['epsilon_decay'], gamma_=c['gamma'], alpha=c['alpha'] )

        toc = time.time()

        elapsed_time = toc - tic

        # Generate wins
        win = qlearning_tabular_test(env, Q, episodes=100, display=False)*100

        new_row = {'episodes':c['episodes'],
                   'epsilon_decay':c['epsilon_decay'],
                   'alpha':   c['alpha'],
                   'gamma': c['gamma'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time}

        results = results.append(new_row, ignore_index=True)

    return results


if __name__ == '__main__':
   
    report = report_sarsa(False)
    print(report)
