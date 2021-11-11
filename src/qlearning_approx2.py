import gym
 
actions_names = {
    'Left': 0,
    'Down': 1,
    'Right': 2, 
    'Up': 3
}
  
env = gym.make("FrozenLake-v0", is_slippery=False, map_name='8x8')
env.reset()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from datetime import datetime 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import pickle
import time


class LinearModel:
	# Linear Regression
	def __init__(self,n_features, action_dim):
		self.W = np.random.randn(n_features, action_dim) / np.sqrt(n_features)
		self.b = np.zeros(action_dim)
		#self.W = np.random.randn(n_features)# / np.sqrt(n_features)
		#self.b = 0

		# momentum terms 
		self.vW = 0
		self.vb = 0

		self.losses = []	

	def predict(self,X): # return an array of 4 values
		if(X.shape[0] > 1 ): #(s,a_0)(s,a_1)(s,a_2)(s,a_3)
			nacts = self.W.shape[1]
			res = np.zeros(nacts)
			for i in range(nacts):
				dot = X[i].dot(self.W[:,i])  + self.b[i]
				res[i] = dot
			return res
		else:
			return X.dot(self.W) + self.b #(s,a_x)

	def sgd(self, X, Y, learning_rate=0.01, momentum=0.9): # Stochastic gradent descent
		Yhat= self.predict(X)
		gW= 2*X.T.dot(Yhat - Y)
		gb= 2*(Yhat-Y).sum(axis=0)

		self.vW = momentum * self.vW - learning_rate * gW
		self.vb = momentum * self.vb - learning_rate * gb 

		self.W +=  self.vW
		self.b +=  self.vb

		mse = np.mean((Yhat-Y)**2)
		self.losses.append(mse)	

	def sgd2(self, X, Y, learning_rate=0.01, momentum=0.9): # Stochastic gradent descent
		assert(len(X.shape)==2)

		num_values=np.prod(Y.shape) # 4 values , 4 actions

		Yhat= self.predict(X)
		gW= 2*X.T.dot(Yhat - Y)/num_values
		gb= 2*(Yhat-Y).sum(axis=0)/num_values

		self.vW = momentum * self.vW - learning_rate * gW
		self.vb = momentum * self.vb - learning_rate * gb 

		self.W +=  self.vW
		self.b +=  self.vb

		mse = np.mean((Yhat-Y)**2)
		self.losses.append(mse)

	def save_weights(self,filepath):
		np.savez(filepath, W=self.W, b=self.b)

	def load_weights(self,filepath):
		npz = np.load(filepath)
		self.W = npz['W']
		self.b = npz['b']
		
def make_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
  
def load(name):
	model.load_weights(name)

def save(name):
	model.save_weights(name)
  
def _next_position(row, col, action):
    row_next = -1
    col_next = -1
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

def getFeatures(state, action):
	n_rows, n_cols = (8, 8)

	row, col = state // n_rows, state % n_cols

	row_next, col_next = _next_position(row, col, action)

	row, col = row / (n_rows - 1), col / (n_cols - 1)
	state_features = np.array([row, col, row * col,
							row**2, col**2,
							row**3, col**3,
							row**4, col**4])
	
	row_next, col_next = row_next / (n_rows - 1), col_next / (n_cols - 1)
	action_features = np.array([row_next, col_next, row_next * col_next,
							row_next**2, col_next**2,
							row_next**3, col_next**3,
							row_next**4, col_next**4])

	features = np.concatenate(
		[[1.0], state_features, action_features])
 
	#features = np.reshape(features,(1, features.size))
	return features
	
def getFeatures2(state):
	n_rows, n_cols = (8, 8)

	row, col = state // n_rows, state % n_cols

	row, col = row / (n_rows - 1), col / (n_cols - 1)
	state_features = np.array([row, col, row * col,
							row**2, col**2,
							row**3, col**3,
							row**4, col**4])

	features = np.concatenate(
		[[1.0], state_features])
 
	#features = np.reshape(features,(1, features.size))
	return features

def FindAction2(state, actions):
	rand = np.random.rand()
	if rand <= epsilon:
		return np.random.choice(action_size) # return random action with probability epsilon
	
	action_values = np.zeros((action_size, n_features))
	
	# get features of the (s,a0) (s, a1) (s, a2) (s, a3)
	for action in actions:
		x = getFeatures(state, action)
		action_values[action] = x
	
	# predict set of features per action with the weight matrix
	qvalues = model.predict(action_values) 
	
	return np.argmax(qvalues[0]) 


def train(statef, action, reward, next_statef, done, epsilon):
	if done:
		target = reward
	else: 
		# compute the official one-step Bellman backup updates 
		target = reward + gamma * np.amax(model.predict(next_statef), axis = 1) # Q-Learning			
	
	target_full = model.predict(statef) # Get the values based on the old parameters W,b
	target_full[0,action] = target # update the entry of the corresponding action # this (1,4)

	model.sgd2(statef, target_full) # Stochastic gradient descent. Run one training step and update W, b

	if epsilon > epsilon_min: # decrease the probability of exploration
		epsilon *= epsilon_decay
	
	return target
	
def play_one_episode( env, is_train, stats, e):

    #state = scaler.transform([[state]]) # scale the state vector
    state = env.reset() # get the initial state 
    actions = np.arange(number_actions)
    done = False

    action = FindAction2(state, actions)   
    statef = getFeatures(state, action)    

    while not done:    
        next_state, reward, done, info = env.step(action)
        action_next = FindAction2(next_state, actions)
        next_statef = getFeatures(next_state, action_next)

        next_statef = np.reshape(next_statef,(1, next_statef.size))
        statef = np.reshape(statef,(1, next_statef.size))
        #next_state = scaler.transform([[next_state]]) # scale the next state
        if is_train == 'train': # if the mode is training
            train(statef, action, reward, next_statef, done, epsilon) # Q-Learning with states' aggregation
        
        statef = next_statef # got to next state
        action = action_next

        stats[e] += reward  
		

state_size = env.observation_space.n  # env.state_dimension # initialize state dimension

#getFeatures
n_features = 19

#getFeatures2 (only state features)
#n_features = 9
action_size = env.action_space.n  # initialize actions dimension


momentum = 0.9      # 'momentum': [0.5, 0.7, 0.9]
learning_rate = 0.01    # 'learning_rate': [0.1, 0.01, 0.001]
gamma = 0.95 # discount factor  'gamma': [1, 0.9, 0.1]
epsilon = 0.5 # exploration     'epsilon':[0.5, 0.1]

epsilon_min = 0.01
epsilon_decay = 0.995

number_actions = env.nA
actions = np.arange(number_actions)

num_episodes= 20000 # epochs
stats = np.zeros(num_episodes)
mode = 'train'
model = LinearModel(n_features, action_size)

models_folder = 'linear_rl_model' # to store the Q-model prarameters
rewards_folder = 'linear_rl_rewards' # to store the values of episodes
make_directory(models_folder)
make_directory(rewards_folder)

np.random.seed(2)
env.seed(2)

num_episodes = 5000

mode ='train'

def start(comb, learning_rate, gamma, epsilon, momentum):
    momentum = momentum
    learning_rate = learning_rate
    gamma = gamma
    epsilon = epsilon

    if mode == 'test':
        load(f'{models_folder}/linear'+str(comb)+'.npz')

    for e in range(num_episodes):
        t0 = datetime.now()
        play_one_episode(env, 'train', stats, e)
        dt = datetime.now() - t0

        if stats[e] > 0.0:
            print(f"Win episode: {e +1}/{num_episodes}, rewards: {stats[e]:.2f}, duration: {dt}")
            env.render()
            time.sleep(0.1) 
        
    save(f'{models_folder}/linear'+str(comb)+'.npz')

    plt.plot(model.losses)
    plt.title("Losses of the "+mode+" model")
    plt.show()

    plt.plot(stats)
    plt.title("Rewards "+mode+" model")
    plt.show()

    return dt, stats[e].sum()
	
param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'gamma': [0.85, 0.9, 0.95, 1.0], 'epsilon':[0.5, 0.1], 'momentum': [0.5, 0.7, 0.9]}
    
results = pd.DataFrame(columns=['c', 'learning_rate', 'gamma', 'epsilon', 'momentum', 'win/loss (%)', 'elapsed time (s)'])

comb = 0    
for c in ParameterGrid(param_grid):
    print("Combination ", comb)
    elapsed_time, wins = start(comb, **c)

    new_row = { 'c' : comb,
                   'learning_rate': c['learning_rate'],
                   'gamma': c['gamma'],
                   'epsilon': c['epsilon'],
                   'momentum': c['momentum'],
                   'win/loss (%)': wins,
                   'elapsed time (s)': elapsed_time}

    results = results.append(new_row, ignore_index=True) 
    comb += 1          