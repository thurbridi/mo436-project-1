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


def feature_function(env, state, action):
    row, col = state // 4, state % 4
    row, col = int(row / 7), int(col / 7)

    state_prox = env.P[state][action][0][1]
    row_prox, col_prox = state_prox // 4, state_prox % 4
    row_prox, col_prox = int(row_prox / 7), int(col_prox / 7)

    #features = np.array([1, row, col, row**2, col**2], dtype='float64')
    features = np.zeros(64)
    features[state] = 1

    action_features = np.zeros(64, dtype=np.float64)
    action_features[state_prox] = 1

    #if state == 63:
    #    action_features[state_prox] = 0

    features = np.concatenate([features, action_features])


    return features



def linear_regression(x, w):
    return np.dot(w, x)


def choose_action(env, s, actions, w, epsilon):
    action_values = np.zeros(len(actions), dtype=np.float64)
    for action in actions:
        x = feature_function(env, s, action)
        action_values[action] = linear_regression(x, w)

    if np.random.rand() < epsilon:
        selected = np.random.choice(len(actions))
    else:
        selected = np.random.choice(np.argwhere(
            action_values == np.max(action_values)).ravel())

    return selected


def sarsa_lambda_approx(env,  episodes=1000, discount=0.9, alpha=0.01, trace_decay=0.9,
                        epsilon=0.1):
    number_actions = env.nA
    actions = np.arange(number_actions)
    x = feature_function(env, 0, 0)
    n_features = len(x)

    w = np.zeros(n_features) + 0.0001

    stats = np.zeros(episodes)

    for episode in range(episodes):
        aux = 0

        state = env.reset()  # Always state=0

        action = choose_action(env, state, actions, w, epsilon)

        x = feature_function(env, state, action)
        z = np.zeros(n_features)
        q_prev = 0

        for t in itertools.count():
            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action(env, state_next, actions, w, epsilon)
            x_next = feature_function(env, state_next, action_next)

            q = linear_regression(x, w)
            q_next = linear_regression(x_next, w)

            if done and reward == 0:
                reward = -1

            delta = reward + discount * q_next - q

            z = discount * trace_decay * z + \
                (1 - alpha * discount * trace_decay * np.dot(z, x)) * x
            w = w + alpha * (delta + q - q_prev) * z - alpha * (q - q_prev) * x
            # w = w + alpha * delta * x

            q_prev = q_next
            x = x_next
            action = action_next

            stats[episode] += reward

            # env.render()
            if done:
                if reward == 1:
                    reward = 1

                # print("episode, aux", episode, aux, reward)
                # else:
                # print('Episode ended: agent fell in the lake')
                break

    return w, stats

def generate_stats_sarsa_approx(env, w_, episodes=100, discount=0.99, alpha=0.01, trace_decay=0.9,
                 epsilon=0.01, display=False):
    number_actions = env.nA
    actions = np.arange(number_actions)
    x = feature_function(env, 0, 0)
    n_features = len(x)
    w = w_
    win_ = 0

    for episode in range(episodes):
        aux = 0
        state = env.reset()  # Always state=0
        action = choose_action(env, state, actions, w, epsilon)
        x = feature_function(env, state, action)
        z = np.zeros(n_features)
        q_prev = 0

        for t in itertools.count():
            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action(env, state_next, actions, w, epsilon)
            x_next = feature_function(env, state_next, action_next)

            q = linear_regression(x, w)
            q_next = linear_regression(x_next, w)

            q_prev = q_next
            x = x_next
            action = action_next

            if display:
                env.render()

            if done:
                if reward == 1:
                    win_ += 1
                break

    return win_/episodes


def report_sarsa_approx(stochastic):
    if stochastic:
        param_ = {'epsilon': [0.01], 'alpha': [0.001, 0.01, 0.1], 'discount': [0.99], 'trace_decay': [0.01, 0.5, 0.9, 1.0], 'episodes': [15000, 20000]}
    else:
        param_ = {'epsilon': [0.01], 'alpha': [0.001, 0.01, 0.1], 'discount': [0.99], 'trace_decay': [0.01, 0.5, 0.9, 1.0], 'episodes': [15000, 20000]}

    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=['episodes', 'gamma', 'alpha', 'lambda', 'epsilon', 'win/loss (%)', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        #print(c)
        # Reset the seed
        np.random.seed(42)
        random.seed(42)
        env.seed(42)

        tic = time.time()

        # Learn policy
        w, stats = sarsa_lambda_approx(env, c['episodes'], c['discount'], c['alpha'], c['trace_decay'], c['epsilon'])

        toc = time.time()

        elapsed_time = toc - tic

        # Generate wins
        win = generate_stats_sarsa_approx(env, w, 100, c['discount'], c['alpha'], c['trace_decay'], c['epsilon'], False)*100

        new_row = {'episodes':c['episodes'],
                   'gamma':c['discount'],
                   'alpha':   c['alpha'],
                   'lambda': c['trace_decay'],
                   'epsilon': c['epsilon'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time}

        results = results.append(new_row, ignore_index=True)

    return results

def draw_feature_sarsa_lambda_approx(env, w):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,8))

    # Make data.
    X = np.arange(0, 8, 1)
    Y = np.arange(0, 8, 1)
    X, Y = np.meshgrid(X, Y)

    states = np.arange(0, 64, 1)

    for a in range(4):
        Z = np.array([linear_regression(feature_function(env, s, a), w)
                      for s in states])
        Z = Z.reshape(8, 8)

        # Plot the surface.
        ax.plot_surface(X, Y, Z, alpha=0.5, linewidth=0.2, antialiased=True)
        ax.azim = -45
        ax.dist = 10
        ax.elev = 20
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.figure()
    plt.show()


if __name__ == '__main__':

    #report = report_sarsa_approx(False)
    #print(report)
    #start = time.time()

    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    w, stats = sarsa_lambda_approx(env, 20000, 1.0, 0.01, 0.9, 0.1)
    draw_feature_sarsa_lambda_approx(env, w)

    #w, f, scal, stats, _  = sarsa_lambda_approx(env, 5000)

    #print(generate_stats_sarsa_approx(env, w, f, scal))
    #end = time.time()
    #print("Algorithm took: ", end-start)

    #w_ = generate_stats(env, Q, E)


    #plt.plot(stats)
    #plt.show()
