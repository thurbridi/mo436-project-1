import gym
import numpy as np
import sys
import time
import pickle
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
#from utils import plotting


def greedy_policy(Q, epsilon, numberAct):
    def policy_fn(observation):
        policy = np.ones(numberAct, dtype=np.float64) * epsilon / numberAct
        best_action = np.argmax(Q[observation])
        policy[best_action] += (1.0 - epsilon)
        return policy
    return policy_fn


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
                    print("episode, aux", episode, aux, reward)
                    env.render()
                break

            state = next_state
            action = next_action

    return Q, stats

if __name__ == '__main__':
    start = time.time()
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)

    Q, stats = sarsa_lambda(env, 1000)
    end = time.time()
    print("Algorithm took: ", end-start)

    plt.plot(stats)
    plt.show()
    #print(Q)
