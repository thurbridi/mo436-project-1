import gym
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
# from utils import plotting

number_states = 64
number_actions = 4
"""
def feature_function(state, action):
    number_cols = np.sqrt(64)
    row, col = state // number_cols, state % number_cols
    #row, col = row / 7, col / 7

    action_features = np.zeros(4, dtype=np.float64)
    action_features[action] = 1

    features = np.array([1, row, col, row**2, col**2,
                        row**2 * col**2], dtype='float64')

    features = np.concatenate([features, action_features])

    if state == 63:
        return np.zeros(features.shape[0])

    return features
"""

def feature_function(env, state, action):
    row, col = state // 4, state % 4
    row, col = int(row / 7), int(col / 7)

    state_prox = env.P[state][action][0][1]
    row_prox, col_prox = state_prox // 4, state_prox % 4
    row_prox, col_prox = int(row_prox / 7), int(col_prox / 7)

    #features = np.zeros(64, dtype=np.float64)
    features = np.array([1, row, col, row*col,
                         row_prox, col_prox, row_prox*col_prox,
                         row*row_prox, col*col_prox], dtype='float64')

    #action_features = np.zeros(4, dtype=np.float64)
    #action_features[action] = 1

    #features = np.array([1, row, col, row**2, col**2], dtype='float64')

    #features = np.concatenate([features, action_features])

    #if state == 63:
    #    return np.zeros(features.shape[0])

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

    w = np.zeros(n_features) + 0.001

    stats = np.zeros(episodes)

    for episode in range(episodes):
        aux = 0

        state = env.reset()  # Always state=0

        action = choose_action(env, state, actions, w, epsilon)

        """
        if w.max() > 10:
            w = (w- w.min()) / (w.max() - w.min())
        if w.min() < -10:
            w = (w- w.min()) / (w.max() - w.min())
        """
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

            #z = z + x

            stats[episode] += reward
            # env.render()

            z = discount * trace_decay * z + x
            w = w + alpha * z * (reward + discount* q_next - q)
            #delta = reward + discount * q_next - q

            #z = discount * trace_decay * z + \
            #    (1 - alpha * discount * trace_decay * np.dot(z, x)) * x

            #w = w + alpha * (delta + q - q_prev) * z - alpha * (q - q_prev) * x
            ## w = w + alpha * delta * x
            if done:
                if reward == 1:
                    print("episode, aux", episode, aux)
                # else:
                # print('Episode ended: agent fell in the lake')
                break

            q_prev = q_next
            x = x_next
            action = action_next

    return w, stats

def draw_feature_sarsa_lambda_approx(env, w):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(0, 8, 1)
    Y = np.arange(0, 8, 1)
    X, Y = np.meshgrid(X, Y)

    states = np.arange(0, 64, 1)

    env.render()
    for a in range(4):
        Z = np.array([linear_regression(feature_function(env, s, a), w)
                      for s in states])
        Z = Z.reshape(8, 8)

        # Plot the surface.
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.figure()
    plt.plot(stats)
    plt.show()


if __name__ == '__main__':
    start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)

    w, stats = sarsa_lambda_approx(
        env, 10000, alpha=0.1, epsilon=0.1, discount=0.9, trace_decay=0.9)

    end = time.time()
    print("Algorithm took: ", end-start)

    print(w)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

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
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)

    plt.figure()
    plt.plot(stats)
    plt.show()
