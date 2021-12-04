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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import time
import pandas
from sklearn.model_selection import ParameterGrid
import gc


class LinearFunc2(km.FunctionApproximator):
    """ linear function approximator (body only does one-hot encoding) """
    pass

class LinearFunc(km.FunctionApproximator):
    """ linear function approximator (body only does one-hot encoding) """
    def body(self, S):
        one_hot_encoding = keras.layers.Lambda(lambda x: K.one_hot(x, 64))
        return one_hot_encoding(S)

class MLP(km.FunctionApproximator):
    """ multi-layer perceptron with one hidden layer """
    def body(self, S):
        one_hot_encoding = keras.layers.Lambda(lambda x: K.one_hot(x, 64))
        X = keras.layers.Flatten()(one_hot_encoding(S))
        X = keras.layers.Dense(units=64, activation='relu')(X)
        X = keras.layers.Dense(units=20, activation='relu')(X)
        X = keras.layers.Dense(units=4)(X)
        return X

def sac_train(env, num_episodes=200, target_model_sync_period=10, lr=0.01, tau_=1.0, punishment=-0.1):
    #func = MLP(env, learning_rate=lr)
    func = LinearFunc2(env, learning_rate=lr)
    sac = km.SoftActorCritic.from_func(func)
    pi = sac.policy
    G = []

    for ep in range(num_episodes):
        s = env.reset()
        s_ant = s
        for t in itertools.count():
            a = pi(s)
            s_next, r, done, info = env.step(a)

            #Small incentive to keep moving
            if np.array_equal(s_next, s) or np.array_equal(s_next, s_ant):
                r = punishment

            sac.update(s, a, r, done)

            if env.T % target_model_sync_period == 0:
                sac.sync_target_model(tau=tau_)

            if done:
                break

            s_ant =  s
            s = s_next
        G.append(env.G)

    return func, sac, pi, G

def sac_test(env, sac, num_test=10, display=False):
    reward_global = 0
    actions = ['L', 'S', 'R', 'N']
    
    for i in range(num_test):
        s = env.reset()

    if display:
        env.render()

    for t in itertools.count():
        if display:
            print(" v(s) = {:.3f}".format(sac.v_func(s)))

            for i, p in enumerate(km.utils.softmax(sac.policy.dist_params(s))):
                print(" pi({:s}|s) = {:.3f}".format(actions[i], p))

            for i, q in enumerate(sac.q_func1(s)):
                print(" q1(s,{:s}) = {:.3f}".format(actions[i], q))

            for i, q in enumerate(sac.q_func2(s)):
                print(" q2(s,{:s}) = {:.3f}".format(actions[i], q))

        a = sac.policy.greedy(s)
        s, r, done, info = env.step(a)

        if display:
            env.render()

        if done:
            reward_global += r
            break

    return reward_global/num_test


tbdir = datetime.datetime.now().strftime('data/tensorboard/%Y_%m_%d_%H')

def search_params(slippery=False, param=0):
    actions = ['L', 'S', 'R', 'N']

    if slippery:
        env = gym.make('FrozenLake8x8-v0', is_slippery=True)
        epi = 2000
    else:
        env = gym.make('FrozenLake8x8-v0', is_slippery=False)
        epi = 1000

    env = km.wrappers.TrainMonitor(env, tensorboard_dir=tbdir)
    #km.enable_logging()

    if param ==0:
        param_ = {'target_model_sync_period': [10], 'episodes': [epi],
                    'lr': [0.005, 0.001], 'tau':[1.0], 'punishment':[-0.1]}
    elif param==1:
        param_ = {'target_model_sync_period': [5, 10, 25], 'episodes': [epi],
                    'lr': [0.005], 'tau':[1.0], 'punishment':[-0.1]}
    elif param==2:
        param_ = {'target_model_sync_period': [10], 'episodes': [epi],
                    'lr': [0.005], 'tau':[1.0, 0.9, 0.5], 'punishment':[-0.1]}
    elif param==3:
        param_ = {'target_model_sync_period': [10], 'episodes': [epi],
                    'lr': [0.005], 'tau':[1.0], 'punishment':[-0.25,-0.1, -0.01]}
    else:
        param_ = {'target_model_sync_period': [10], 'episodes': [epi],
                    'lr': [0.005], 'tau':[1.0, 0.5], 'punishment':[-0.25, -0.1, -0.01]}

    results = pandas.DataFrame(columns=['model_sync_period', 'episodes', 'lr', 'tau',
                            'punishment', 'reward_train', 'reward', 'elapsed time (s)'])

    for c in ParameterGrid(param_):
        print(c)
        if slippery:
            env = gym.make('FrozenLake8x8-v0', is_slippery=True)
        else:
            env = gym.make('FrozenLake8x8-v0', is_slippery=False)

        gc.collect()  #clean memory form coolab
        np.random.seed(2)
        tf.random.set_seed(2)
        env.seed(2)
        keras.backend.clear_session()

        env = km.wrappers.TrainMonitor(env, tensorboard_dir=tbdir)
        #km.enable_logging()
        

        model_syn_period= c['target_model_sync_period']
        episodes        = c['episodes']
        lr              = c['lr']
        tau_            = c['tau']
        punishment      = c['punishment']

        tic = time.time()
        func, sac, pi, Reward = sac_train(env, episodes , model_syn_period, lr, tau_, punishment)
        toc = time.time()

        elapsed_time = toc - tic
        scores = sac_test(env, sac, num_test=50, display=False)

        new_row = {'model_sync_period': c['target_model_sync_period'],
                    'episodes'        : c['episodes'],
                    'lr'              : c['lr'],
                    'tau'             : c['tau'],
                    'punishment'      : c['punishment'],
                    'reward_train'    : (np.array(Reward).sum()/episodes)*100,
                    'reward'          : scores*100,
                    'elapsed time (s)': elapsed_time}

        results = results.append(new_row, ignore_index=True)
        print("result: ", scores*100, (np.array(Reward).sum()/episodes)*100)

    return results


if __name__ == '__main__':
    result = search_params(False, 0)
    print(result)
