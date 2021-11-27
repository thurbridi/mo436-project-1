import gym
import sys
import os
import time
import pandas
import random
import pickle
import itertools
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import keras_gym as km
import datetime

class LinearFunc(km.FunctionApproximator):
    #Linear function approx (one-shot encoding)
    pass

def sac_train(env, num_episodes=200, target_model_sync_period=10, lr=0.01):
    func = LinearFunc(env, learning_rate=lr)
    sac = km.SoftActorCritic.from_func(func)
    pi = sac.policy
    G = []

    for ep in range(num_episodes):
        s = env.reset()
        for t in itertools.count():
            a = pi(s)
            s_next, r, done, info = env.step(a)

            if done and r==0:
                r = 0

            #Small incentive to keep moving
            if np.array_equal(s_next, s):
                r = -0.01

            sac.update(s, a, r, done)

            if env.T % target_model_sync_period == 0:
                sac.sync_target_model(tau=1.0)

            if done:
                break

            s = s_next
        G.append(env.G)

    return func, sac, pi, G


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v0', is_slippery=False)

    actions = ['L', 'S', 'R', 'N']
    target_model_sync_period = 10
    episodes = 150
    lr = 0.01

    tbdir = datetime.datetime.now().strftime('data/tensorboard/%Y_%m_%d_%H')
    env = km.wrappers.TrainMonitor(env, tensorboard_dir=tbdir)

    #km.enable_logging()

    func, sac, pi, Reward = sac_train(env, episodes , target_model_sync_period, lr)

    s = env.reset()
    env.render()

    actions = ['L', 'S', 'R', 'N']

    for t in itertools.count():
        print(" v(s) = {:.3f}".format(sac.v_func(s)))

        for i, p in enumerate(km.utils.softmax(sac.policy.dist_params(s))):
            print(" pi({:s}|s) = {:.3f}".format(actions[i], p))

        for i, q in enumerate(sac.q_func1(s)):
            print(" q1(s,{:s}) = {:.3f}".format(actions[i], q))

        for i, q in enumerate(sac.q_func2(s)):
            print(" q2(s,{:s}) = {:.3f}".format(actions[i], q))

        a = sac.policy.greedy(s)
        s, r, done, info = env.step(a)

        env.render()

        if done:
            break

    print(Reward)

    """
    Q_sarsa, E_sarsa, stats, _ = sarsa_lambda(
        env, 1000, 0.99, 0.01, 0.5, 0.01, 'accumulate')
    print("WOW")
    time.sleep(3)
    generate_stats_sarsa(env, Q_sarsa, E_sarsa, 1, 0.99,
                         0.01, 0.5, 0.0, 'accumulate', True)
    """
    """
    env = gym.make('FrozenLake8x8-v1', is_slippery=True)
    Q_ssarsa, E_ssarsa, stats, _ = sarsa_lambda(env, 5000, 0.99, 0.01, 0.5, 0.05, 'accumulate')
    print("WOW")
    time.sleep(3)
    generate_stats_sarsa(env, Q_ssarsa, E_ssarsa, 1, 0.99, 0.01, 0.5, 0.0, 'accumulate', True)
    """
