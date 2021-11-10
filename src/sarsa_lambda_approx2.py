##############################################################################
#                   CODE OF ARTHUR                                           #
##############################################################################
import gym
import numpy as np
import os
import time
import pandas
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


action_names = {
    0: 'left',
    1: 'down',
    2: 'right',
    3: 'up'
}


def _next_position(row, col, action):
    if action == 0:
        row_next, col_next = row, col - 1
    elif action == 1:
        row_next, col_next = row + 1, col
    elif action == 2:
        row_next, col_next = row, col + 1
    elif action == 3:
        row_next, col_next = row - 1, col

    row_next = max(0, min(row_next, 7))
    col_next = max(0, min(col_next, 7))

    return row_next, col_next


def feature_function2(state, action):
    n_rows, n_cols = (8, 8)

    row, col = state // n_rows, state % n_cols

    row_next, col_next = _next_position(row, col, action)

    row, col = row / (n_rows - 1), col / (n_cols - 1)
    state_features = np.array([row, col, row * col,
                               row**2, col**2,
                               row**3, col**3,
                               row**4, col**4],
                              dtype=np.float64)

    row_next, col_next = row_next / (n_rows - 1), col_next / (n_cols - 1)
    action_features = np.array([row_next, col_next, row_next * col_next,
                               row_next**2, col_next**2,
                               row_next**3, col_next**3,
                               row_next**4, col_next**4],
                               dtype=np.float64)

    features = np.concatenate(
        [[1.0], state_features, action_features])

    return features


def linear_regression2(x, w):
    return np.dot(w, x)


def choose_action2(s, actions, w, epsilon):
    action_values = np.zeros(len(actions), dtype=np.float64)
    for action in actions:
        x = feature_function2(s, action)
        action_values[action] = linear_regression2(x, w)

    if np.random.rand() < epsilon:
        selected = np.random.choice(len(actions))
    else:
        selected = np.random.choice(np.argwhere(
            action_values == np.max(action_values)).ravel())

    return selected


def sarsa_lambda_approx2(env,  episodes=1000, discount=0.9, alpha=0.01, trace_decay=0.9,
                        epsilon=0.1, verbose=False):
    number_actions = env.nA
    actions = np.arange(number_actions)
    n_features = 19

    w = np.zeros(n_features)

    stats = np.zeros(episodes)

    for episode in range(episodes):
        aux = 0

        state = env.reset()  # Always state=0

        action = choose_action2(state, actions, w, epsilon)

        x = feature_function2(state, action)
        z = np.zeros(n_features)
        q_prev = 0

        for t in itertools.count():
            if t > 150:
                if verbose:
                    print('Early termination: exceeded number of steps')
                break

            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action2(state_next, actions, w, epsilon)
            x_next = feature_function2(state_next, action_next)

            q = linear_regression2(x, w)
            q_next = linear_regression2(x_next, w)

            delta = reward + discount * q_next - q

            z = discount * trace_decay * z + \
                (1 - alpha * discount * trace_decay * np.dot(z, x)) * x
            w = w + alpha * (delta + q - q_prev) * z - alpha * (q - q_prev) * x

            q_prev = q_next
            x = x_next
            action = action_next

            stats[episode] += reward

            if verbose:
                if episode == episodes - 1:
                    env.render()
                    time.sleep(0.1)

            if done:
                if verbose:
                    if reward == 1:
                        print("episode, aux", episode, aux, reward)
                break

    return w, stats


def generate_stats_sarsa_approx2(env, w, episodes=100,
                                epsilon=0.0, display=False):
    number_actions = env.nA
    actions = np.arange(number_actions)
    win_ = 0

    for episode in range(episodes):
        aux = 0
        state = env.reset()  # Always state=0
        action = choose_action2(state, actions, w, epsilon)
        x = feature_function2(state, action)

        for t in itertools.count():
            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action2(state_next, actions, w, epsilon)
            x_next = feature_function2(state_next, action_next)

            x = x_next
            action = action_next

            if display:
                #os.system('clear')
                env.render()
                #time.sleep(1)

            if done:
                if reward == 1:
                    win_ += 1
                if display:
                    #os.system('clear')
                    env.render()
                break

    return win_/episodes


def report_sarsa_approx2(stochastic):
    if stochastic:
        param_ = {'epsilon': [0.01], 'alpha': [1e-2, 1e-3, 1e-5], 'discount': [
            1.0], 'trace_decay': [0.0, 0.4, 0.6, 1.0], 'episodes': [1500, 2000]}
    else:
        param_ = {'epsilon': [0.01], 'alpha': [1e-2, 1e-3, 1e-5], 'discount': [
            1.0], 'trace_decay': [0.0, 0.4, 0.6, 1.0], 'episodes': [1500, 2000]}

    env = gym.make('FrozenLake8x8-v1', is_slippery=stochastic)

    results = pandas.DataFrame(columns=[
                               'episodes', 'gamma', 'alpha', 'lambda', 'epsilon', 'win/loss (%)', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        # print(c)
        # Reset the seed
        np.random.seed(42)
        random.seed(42)
        env.seed(42)

        tic = time.time()

        # Learn policy
        w, stats = sarsa_lambda_approx2(
            env, c['episodes'], c['discount'], c['alpha'], c['trace_decay'], c['epsilon'])

        toc = time.time()

        elapsed_time = toc - tic

        # Generate wins
        win = generate_stats_sarsa_approx2(
            env, w, 100, c['epsilon'], False)*100

        new_row = {'episodes': c['episodes'],
                   'gamma': c['discount'],
                   'alpha':   c['alpha'],
                   'lambda': c['trace_decay'],
                   'epsilon': c['epsilon'],
                   'win/loss (%)': win,
                   'elapsed time (s)': elapsed_time}

        results = results.append(new_row, ignore_index=True)

    return results


def plot_action_value2(w, grid_shape=(8, 8)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,8))
    rows, cols = grid_shape

    # Make data.
    X = np.arange(0, rows, 1)
    Y = np.arange(0, cols, 1)
    X, Y = np.meshgrid(X, Y)

    states = np.arange(0, rows * cols, 1)
    for a in range(4):
        Z = np.array([linear_regression2(feature_function2(s, a), w)
                      for s in states])
        Z = Z.reshape(rows, cols)

        ax.azim = -45
        ax.dist = 10
        ax.elev = 30
        # Plot the surface.
        ax.plot_surface(X, Y, Z, linewidth=0,
                        antialiased=False, label=action_names[a])
    ax.set_xlabel("Row")
    ax.set_ylabel("Column")
    ax.set_zlabel("Q(s,a,w)")

    return fig


if __name__ == '__main__':
    """
    np.random.seed(777)
    start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)

    w, stats = sarsa_lambda_approx2(
        env, 10000, alpha=1e-5, epsilon=0.0, discount=1.0, trace_decay=0.6)

    end = time.time()

    win_ratio = generate_stats_sarsa_approx2(env, w)

    print("Algorithm took: ", end-start)

    print(w)

    print(f'Win ratio: {win_ratio}')

    plot_action_value2(w)

    plt.figure()
    plt.plot(stats)
    plt.show()
    """
    """
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    w_dlfsarsa2, stats = sarsa_lambda_approx2(env, 1500, alpha=1e-5, epsilon=0.01, discount=1.0, trace_decay=0.6)
    print("WOW")
    time.sleep(3)
    generate_stats_sarsa_approx2(env, w_dlfsarsa2, episodes=1, display=True)
    """
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)
    w_slfsarsa2, stats = sarsa_lambda_approx2(env, 5000, alpha=1e-2, epsilon=0.01, discount=1.0, trace_decay=0.0)
    print("WOW")
    time.sleep(3)
    generate_stats_sarsa_approx2(env, w_slfsarsa2, episodes=1, display=True)
