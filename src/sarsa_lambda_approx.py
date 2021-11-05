import gym
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
#from utils import plotting


def feature_function(state, action):
    feature = np.zeros(6, dtype=np.float64)
    if action == 0:
        feature[1] = 1.0
    elif action ==1:
        feature[2] = 1.0
    elif action == 2:
        feature[3] = 1.0
    else:
        feature[4] = 1.0
    feature[0] = float(state) / 64.0
    feature[5] = 1.0

    return feature


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
    n_features = 6

    w = np.zeros(n_features) + 0.001

    stats = np.zeros(episodes)

    for episode in range(episodes):
        aux = 0

        state = env.reset()  # Always state=0

        action = choose_action(state, actions, w, epsilon)

        x = feature_function(state, action)
        z = np.zeros(n_features)
        q_prev = 0

        for t in itertools.count():
            aux += 1

            state_next, reward, done, _ = env.step(action)
            action_next = choose_action(state_next, actions, w, epsilon)
            x_next = feature_function(state_next, action_next)

            q = linear_regression(x, w)
            q_next = linear_regression(x_next, w)

            if reward == 0  and done:
                reward = -1

            delta = reward + discount * q_next - q

            z = discount * trace_decay * z + \
                (1 - alpha * discount * trace_decay * np.dot(z, x)) * x

            w = w + alpha * (delta + q - q_prev) * z - alpha * (q - q_prev) * x

            q_prev = q_next
            x = x_next
            action = action_next

            stats[episode] += reward

            if done:
                if reward == 1:
                    print("episode, aux", episode, aux, reward)
                    #env.render()
                break

    return w, stats


if __name__ == '__main__':
    start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)

    w, stats = sarsa_lambda_approx(
        env, 10000, alpha=0.1, epsilon=0.1)

    end = time.time()
    print("Algorithm took: ", end-start)

    plt.plot(stats)
    plt.show()
    print(w)
