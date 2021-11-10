import gym
 
actions_names = {
    'Left': 0,
    'Down': 1,
    'Right': 2, 
    'Up': 3
}
  
env = gym.make("FrozenLake-v0", is_slippery=False, map_name='8x8')
env.reset()

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import pickle
import time


class LinearModel:
	# Linear Regression
	def __init__(self,n_features, action_dim):
		#self.W = np.random.randn(n_features, action_dim) / np.sqrt(n_features)
		#self.b = np.zeros(action_dim)
		self.W = np.random.randn(n_features) / np.sqrt(n_features)
		self.b = 0

		# momentum terms 
		self.vW = 0
		self.vb = 0

		self.losses = []	

	def predict(self,X):		
		return X.dot(self.W)

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

	def save_weights(self,filepath):
		np.savez(filepath, W=self.W, b=self.b)

	def load_weights(self,filepath):
		npz = np.load(filepath)
		self.W = npz['W']
		self.b = npz['b']

om sklearn.preprocessing import StandardScaler
from datetime import datetime 
import numpy as np

state_size = env.observation_space.n  # env.state_dimension # initialize state dimension
n_features = 19
action_size = env.action_space.n  # initialize actions dimension

momentum=0.9
learning_rate=0.01
gamma = 0.95 # discount factor
epsilon = 0.5 # exploration
epsilon_min = 0.01
epsilon_decay = 0.995

number_actions = env.nA
actions = np.arange(number_actions)

num_episodes= 5000 # epochs
stats = np.zeros(num_episodes)
mode = 'test'
model = LinearModel(n_features, action_size)

def make_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)
  
def load(name):
	model.load_weights(name)

def save(name):
	model.save_weights(name)
  
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

def FindActio2n(state, actions):
	rand = np.random.rand()
	if rand <= epsilon:
		return np.random.choice(action_size) # return random action with probability epsilon
	
	action_values = np.zeros((action_size, n_features))
	
	for action in actions:
		x = getFeatures(state, action)
		action_values[action] = x
	
	qvalues = model.predict(action_values) 
 
	return np.argmax(qvalues)
 
def FindAction(state, actions):
	rand = np.random.rand()
	if rand <= epsilon:
		return np.random.choice(action_size) # return random action with probability epsilon
	
	qvalues = np.zeros(action_size)
	
	for action in actions:
		x = getFeatures(state, action)
		qvalues[action] = model.predict(x)
 
	return  np.argmax(qvalues)
 

def train(state, action, reward, next_state, done, epsilon):
	if done:
		target = reward
	else:  
		preds = np.zeros(action_size)
		for a in range(action_size):
			preds[a] = model.predict(next_state)	
   
		target = reward + gamma * np.amax(preds) # Q-Learning
		# compute the official one-step Bellman backup updates	

	model.sgd(state, target) # Stochastic gradient descent. Run one training step and update W, b
	if epsilon > epsilon_min: # decrease the probability of exploration
		epsilon *= epsilon_decay
	
	return target
	
def play_one_episode( env, is_train, stats, e):

    #state = scaler.transform([[state]]) # scale the state vector
    state = env.reset() # get the initial state 
    actions = np.arange(number_actions)
    done = False

    action = FindAction(state, actions)   
    statef = getFeatures(state, action) 
   
    aux = 0

    while not done:
                
        next_state, reward, done, info = env.step(action)

        action_next = FindAction(next_state, actions)
        next_statef = getFeatures(next_state, action_next)

        #next_state = scaler.transform([[next_state]]) # scale the next state
        if is_train == 'train': # if the mode is training
            train(statef, action, reward, next_statef, done, epsilon) # Q-Learning with states' aggregation
        
        statef = next_statef # got to next state
        action = action_next

        stats[e] += reward
        
        if reward == 1:
            env.render()
            time.sleep(0.1)    

        aux += 1
    return reward
	
models_folder = 'linear_rl_model' # to store the Q-model prarameters
rewards_folder = 'linear_rl_rewards' # to store the values of episodes
make_directory(models_folder)
make_directory(rewards_folder)

np.random.seed(42)
env.seed(42)

if mode == 'test':
	# remake the env with the test data
	#env = StockEnv(test_data, initial_investment)
	load(f'{models_folder}/linear.npz')

for e in range(num_episodes):
	t0 = datetime.now()
	r = play_one_episode(env, 'train', stats, e)
	dt = datetime.now() - t0

	#if e % 100 == 0:
		#print(f"episode: {e +1}/{num_episodes}, reward: {r:.2f}, duration: {dt}")
	if r > 0.0:
		print(f"Win episode: {e +1}/{num_episodes}, reward: {r:.2f}, duration: {dt}")
	save(f'{models_folder}/linear.npz')

plt.plot(model.losses)
plt.title("Losses of the "+mode+" model")
plt.show()

plt.plot(stats)
plt.title("Rewards "+mode+" model")
plt.show()