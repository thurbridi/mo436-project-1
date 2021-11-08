import gym
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
# from utils import plotting

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


def feature_function(state, action):
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


def linear_regression(x, w):
    return np.dot(w, x)


def choose_action(s, actions, w, epsilon):
    action_values = np.zeros(len(actions), dtype=np.float64)
    for action in actions:
        x = feature_function(s, action)
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
    n_features = 19

    w = np.zeros(n_features)

    stats = np.zeros(episodes)

    for episode in range(episodes):
        aux = 0

        state = env.reset()  # Always state=0

        action = choose_action(state, actions, w, epsilon)

        x = feature_function(state, action)
        z = np.zeros(n_features)
        q_prev = 0

        for t in itertools.count():
            if t > 150:
                print('Early termination: exceeded number of steps')
                break

            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action(state_next, actions, w, epsilon)
            x_next = feature_function(state_next, action_next)

            q = linear_regression(x, w)
            q_next = linear_regression(x_next, w)

            delta = reward + discount * q_next - q

            z = discount * trace_decay * z + \
                (1 - alpha * discount * trace_decay * np.dot(z, x)) * x
            w = w + alpha * (delta + q - q_prev) * z - alpha * (q - q_prev) * x

            q_prev = q_next
            x = x_next
            action = action_next

            stats[episode] += reward

            if episode == episodes - 1:
                env.render()
                time.sleep(0.1)

            if done:
                if reward == 1:
                    print("episode, aux", episode, aux, reward)
                break

    return w, stats


def plot_action_value(w, grid_shape=(8, 8)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    rows, cols = grid_shape

    # Make data.
    X = np.arange(0, rows, 1)
    Y = np.arange(0, cols, 1)
    X, Y = np.meshgrid(X, Y)

    states = np.arange(0, rows * cols, 1)
    for a in range(4):
        Z = np.array([linear_regression(feature_function(s, a), w)
                      for s in states])
        Z = Z.reshape(rows, cols)

        # Plot the surface.
        ax.plot_surface(X, Y, Z, linewidth=0,
                        antialiased=False, label=action_names[a])

    return fig


if __name__ == '__main__':
    np.random.seed(777)
    start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)

    w, stats = sarsa_lambda_approx(
        env, 1000, alpha=1e-5, epsilon=0.1, discount=1, trace_decay=0.4)

    end = time.time()
    print("Algorithm took: ", end-start)

    print(w)

    plot_action_value(w)

    plt.figure()
    plt.plot(stats)
    plt.show()
