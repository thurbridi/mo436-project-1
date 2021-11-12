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
	def __init__(self,n_features, action_dim, learning_rate=0.01, momentum=0.9):
		self.W = np.random.randn(n_features)
		self.b = 0

		# parameters
		self.learning_rate = learning_rate
		self.momentum = momentum

		# momentum terms 
		self.vW = 0
		self.vb = 0

		self.losses = []	

	def predict(self,X):		
		return X.dot(self.W) + self.b

	def sgd(self, X, Y):
		Yhat= self.predict(X)
		
		gW= 2*X.T.dot(Yhat - Y)
		gb= 2*(Yhat-Y)

		self.vW = self.momentum * self.vW - self.learning_rate * gW
		self.vb = self.momentum * self.vb - self.learning_rate * gb 

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
	
	return features


def FindAction(state, actions, model, epsilon):
	rand = np.random.rand()
	if rand <= epsilon:
		return np.random.choice(action_size) # return random action with probability epsilon
	
	qvalues = np.zeros(action_size)
	
	for action in actions:
		x = getFeatures(state, action)
		qvalues[action] = model.predict(x)
 
	return  np.argmax(qvalues)
 

def train(statef, action, reward, next_state, done, epsilon, gamma, model):
	if done:
		target = reward
	else:  
		preds = np.zeros(action_size)
		for a in range(action_size):
			preds[a] = model.predict(next_state)	
   
		# compute the official one-step Bellman backup updates	
		target = reward + gamma * np.amax(preds) # Q-Learning
		
	# Stochastic gradient descent. Run one training step and update W, b
	model.sgd(statef, target)
	
	# decrease the probability of exploration
	if epsilon > epsilon_min: 
		epsilon *= epsilon_decay
		
def play_one_episode( env, rewards, gamma, epsilon, model, e):
    
    state = env.reset()
    actions = np.arange(number_actions)
    done = False

    action = FindAction(state, actions, model, epsilon)   
    statef = getFeatures(state, action) 

    while not done:                
        next_state, reward, done, info = env.step(action)

        action_next = FindAction(next_state, actions, model, epsilon)
        next_statef = getFeatures(next_state, action_next)

        train(statef, action, reward, next_statef, done, epsilon, gamma, model) # Q-Learning with states' aggregation
        
        statef = next_statef
        action = action_next

        rewards[e] += reward
    
    if done==True and reward==0.0:
        rewards[e] += -1
		
from sklearn.model_selection import ParameterGrid
from datetime import datetime 
import numpy as np

# to store the Q-model prarameters
models_folder = 'linear_rl_model' 
make_directory(models_folder)

state_size = env.observation_space.n

n_features = 19
#n_features = 9 # (only state features)

action_size = env.action_space.n

epsilon_min = 0.01
epsilon_decay = 0.995

number_actions = env.nA
actions = np.arange(number_actions)

num_episodes= 20000
mode = 'train'


def grid_search_approximator(comb, learning_rate, gamma, epsilon, momentum):
    momentum = momentum
    learning_rate = learning_rate
    gamma = gamma
    epsilon = epsilon
    rewards = np.zeros(num_episodes)

    model = LinearModel(n_features, action_size, learning_rate= learning_rate, momentum= momentum)

    for e in range(num_episodes):
        t0 = datetime.now()
        play_one_episode(env, rewards, gamma, epsilon, model, e)
        dt = datetime.now() - t0

        #if rewards[e] > 0.0:
            #print(f"Win episode: {e +1}/{num_episodes}, reward: {rewards[e]:.2f}, duration: {dt}")

    #save(f'{models_folder}/linear'+str(comb)+'.npz')

    return dt, rewards, model.losses
	
param_grid = {'learning_rate': [0.01, 0.001, 0.0001], 'gamma': [0.85, 0.95, 1.0], 'epsilon':[0.5, 0.1], 'momentum': [ 0.7, 0.9]}
    
table = pd.DataFrame(columns=['c', 'learning_rate', 'gamma', 'epsilon', 'momentum', 'wins', 'avg loss', 'elapsed time (s)'])

comb = 0
rewards_per_model = []
loss_per_model = []

for c in ParameterGrid(param_grid):
    print("Combination ", comb)

    elapsed_time, rewards, losses = grid_search_approximator(comb, **c)

    rewards_per_model.append(rewards)
    loss_per_model.append(losses)
    avr_loss = np.mean(losses)
    wins = (rewards == 1).sum()

    new_row = { 'c' : comb,
                'learning_rate': c['learning_rate'],
                'gamma': c['gamma'],
                'epsilon': c['epsilon'],
                'momentum': c['momentum'],
                'wins': wins,
                'avg loss': avr_loss,
                'elapsed time (s)': elapsed_time}

    table = table.append(new_row, ignore_index=True) 
    comb += 1
    print(new_row)

print(table)	

for i in range(len(loss_per_model)):
    lis = loss_per_model[i]
    plt.plot(lis)
    plt.title("Loss "+str(i))
    plt.show()
	
for i in range(len(rewards_per_model)):
    lis = rewards_per_model[i]
    plt.plot(lis)
    plt.title("Rewards "+str(i))
    plt.show()