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
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler


def policy_sarsa_approx(x, epsilon, numberAct):
    def policy_fn(observation, w):
        p = np.random.random()
        if p < (1 - epsilon):
            Q = np.asarray([np.dot(w, x(observation, a)) for a in range(numberAct)])
            return np.random.choice(np.argwhere(Q == np.max(Q)).ravel())
        else:
            return np.random.randint(0, numberAct)

    return policy_fn


def sarsa_lambda_approx(env,  episodes=1000, discount=0.99, alpha=0.01, trace_decay=0.9,
                 epsilon=0.01, type='accumulate'):
    number_actions = env.nA
    number_states = env.observation_space.n

    featurizer = RBFSampler(gamma=1, n_components=50, random_state=1)
    scaler = StandardScaler()

    # Collect observations
    X = np.asarray([[np.random.randint(0, number_states), np.random.randint(0, number_actions)] for x in range(30000)])

    # Fit the feature function vector
    scaler.fit(X)
    featurizer.fit(scaler.transform(X))

    # Generate the feature funtion
    q = lambda state, action: featurizer.transform(scaler.transform(np.asarray([[state, action]])))[0]

    # Get the feature vector shape
    m = q(0, 0).shape

    # Initialize weight vector
    w = np.zeros(m[0]) + 0.001

    #Initialize Trace
    E = defaultdict(lambda: np.zeros(number_actions))
    aux = 0
    policy = policy_sarsa_approx(q, epsilon, number_actions)

    stats = np.zeros(episodes)
    rewards = [0.0]
    win_ = False

    for episode in range(episodes):
        aux = 0
        state = env.reset() #Always state=0
        action = policy(state, w)
        for t in itertools.count():
            aux += 1
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, w)

            if reward == 0  and done:
                reward = -1

            Q_next = np.dot(w, q(next_state,next_action) )
            Q = np.dot(w, q(state,action) )
            delta = reward + discount*Q_next - Q

            w = w + alpha* delta * q(state,action)

            stats[episode] += reward
            """
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
            """
            if done:
                if reward == 1:
                    print(episode, aux, reward)
                    env.render()
                    win_ = True
                break

            state = next_state
            action = next_action

    return w, featurizer, scaler, stats, win_

def generate_stats_sarsa_approx(env, w_, f, scal, episodes=5000, discount=0.99, alpha=0.01, trace_decay=0.9,
                 epsilon=0.01, type='accumulate', display=False):
    number_actions = env.nA
    number_states = env.observation_space.n

    featurizer = f
    scaler = scal

    # Generate the feature funtion
    q = lambda state, action: featurizer.transform(scaler.transform(np.asarray([[state, action]])))[0]

    # Get the feature vector shape
    m = q(0, 0).shape

    # Initialize weight vector
    w = w_

    #Initialize Trace
    E = defaultdict(lambda: np.zeros(number_actions))
    aux = 0
    policy = policy_sarsa_approx(q, epsilon, number_actions)

    stats = np.zeros(episodes)
    rewards = [0.0]
    win_ = 0

    for episode in range(100):
        aux = 0
        state = env.reset() #Always state=0
        action = policy(state, w)
        for t in itertools.count():
            aux += 1
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, w)

            if done:
                if reward == 1:
                    win_ += 1
                break

            if display:
                env.render()

            state = next_state
            action = next_action

    return win_/100


def report_sarsa_approx(stochastic):
    if stochastic:
        param_ = {'type': ['accumulate', 'replace'], 'epsilon': [0.01, 0.05, 0.1], 'alpha': [0.01], 'discount': [1.0, 0.99, 0.9], 'trace_decay': [0.9], 'episodes': [5000, 1000]}
    else:
        param_ = {'type': ['accumulate', 'replace'], 'epsilon': [0.01, 0.05, 0.1], 'alpha': [0.01], 'discount': [1.0, 0.99, 0.9], 'trace_decay': [0.9], 'episodes': [5000, 1000]}

    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=['type', 'epsilon', 'alpha', 'discount', 'trace_decay', 'episodes', 'win/loss (%)', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        print(c)
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
                   'discount':c['discount'],
                   'trace_decay': c['trace_decay'],
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

    w, f, scal, stats, _  = sarsa_lambda_approx(env, 5000)

    print(generate_stats_sarsa_approx(env, w, f, scal))
    #end = time.time()
    #print("Algorithm took: ", end-start)

    #w_ = generate_stats(env, Q, E)


    plt.plot(stats)
    plt.show()
