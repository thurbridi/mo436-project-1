import gym
import numpy as np
import sys
import time
import pandas
import random
import pickle
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
#from utils import plotting


def greedy_policy(Q, epsilon, numberAct):
    def policy_fn(observation):
        policy = np.ones(numberAct, dtype=np.float64) * epsilon / numberAct
        best_action = np.argmax(Q[observation])
        policy[best_action] += (1.0 - epsilon)
        return policy
    return policy_fn

def print_state_values_sarsa(Q, size=8):
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

def print_policy_sarsa(Q, size=8):

    actions_names = ['l', 's', 'r', 'n']

    print("\n\t\t Policy/Actions")

    print("------------------------------------------------")
    for i in range(size):
        for j in range(size):
            k = i*size + j
            q = np.argmax(Q[k])
            print("  %s  |" % actions_names[q], end="")
        print("\n------------------------------------------------")



def sarsa_lambda(env,  episodes=1000, discount=0.9, alpha=0.01, trace_decay=0.9,
                 epsilon=0.1, type='accumulate'):
    number_actions = env.nA
    #Initialize Q(s,a) with 0
    Q = defaultdict(lambda: np.zeros(number_actions))

    #Initialize Trace
    E = defaultdict(lambda: np.zeros(number_actions))
    aux = 0
    policy = greedy_policy(Q, epsilon, number_actions)

    stats = np.zeros(episodes)
    rewards = [0.0]
    win_ = False

    for episode in range(episodes):
        aux = 0
        state = env.reset() #Always state=0
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():
            aux += 1
            next_state, reward, done, _ = env.step(action)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            if reward == 0  and done:
                reward = -1
            delta = reward + discount*Q[next_state][next_action] - Q[state][action]

            E[state][action] += 1
            stats[episode] += reward

            for s, _ in Q.items():
                Q[s][:] += alpha * delta * E[s][:]
                if type == 'accumulate':
                    E[s][:] *= trace_decay * discount
                elif type == 'replace':
                    if s== state:
                        E[s][:] = 1
                    else:
                        E[s][:] *= discount * trace_decay

            if done:
                if reward == 1:
                    win_ = True
                break

            state = next_state
            action = next_action

    return Q, E, stats, win_

def generate_stats_sarsa(env, Q_, E_, episodes=1000, discount=0.9, alpha=0.01, trace_decay=0.9,
                 epsilon=0.1, type='accumulate', display=True):
    number_actions = env.nA
    #Initialize Q(s,a) with 0
    Q = Q_

    #Initialize Trace
    E = E_
    aux = 0
    policy = greedy_policy(Q, epsilon, number_actions)

    stats = np.zeros(episodes)
    rewards = [0.0]
    win_ = 0

    for episode in range(100):
        aux = 0
        state = env.reset() #Always state=0
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        for t in itertools.count():
            aux += 1
            next_state, reward, done, _ = env.step(action)
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            if done:
                if reward == 1:
                    win_ += 1
                break

            if display:
                env.render()

            state = next_state
            action = next_action

    return win_/100


def report_sarsa(stochastic):
    if stochastic:
        param_ = {'type': ['accumulate', 'replace'], 'epsilon': [0.01, 0.05, 0.1], 'alpha': [0.01], 'discount': [1.0, 0.99, 0.9], 'trace_decay': [0.9], 'episodes': [1000, 5000]}
    else:
        param_ = {'type': ['accumulate', 'replace'], 'epsilon': [0.01, 0.05, 0.1], 'alpha': [0.01], 'discount': [1.0, 0.99, 0.9], 'trace_decay': [0.9], 'episodes': [1000, 5000]}

    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=['type', 'epsilon', 'alpha', 'gamma', 'lambda', 'episodes', 'win/loss (%)', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        #print(c)
        # Reset the seed
        np.random.seed(42)
        random.seed(42)
        env.seed(42)

        tic = time.time()

        # Learn policy
        Q, E, stats, _ = sarsa_lambda(env, c['episodes'], c['discount'], c['alpha'], c['trace_decay'], c['epsilon'], c['type'])

        toc = time.time()

        elapsed_time = toc - tic

        # Generate wins
        win = generate_stats_sarsa(env, Q, E, c['episodes'], c['discount'], c['alpha'], c['trace_decay'], c['epsilon'], c['type'], False)*100

        new_row = {'type':    c['type'],
                   'epsilon': c['epsilon'],
                   'alpha':   c['alpha'],
                   'gamma':c['discount'],
                   'lambda': c['trace_decay'],
                   'episodes':c['episodes'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time}

        results = results.append(new_row, ignore_index=True)

    return results

if __name__ == '__main__':
    #report = report_sarsa(False)
    #print(report)
    #start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)

    Q, E, stats, _  = sarsa_lambda(env, 1000)
    #end = time.time()
    #print("Algorithm took: ", end-start)

    #w_ = generate_stats(env, Q, E)


    plt.plot(stats)
    plt.show()
