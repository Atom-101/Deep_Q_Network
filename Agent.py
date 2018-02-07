import random
#import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import Environment as env
EPISODES = 1000


class DQNAgent(object):
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)#max length of memory is set to 2000
		self.gamma = 0.95	# discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.C = 16
		self.counter = 0
		self.model = self._build_model()
		self.model2 = self._build_model()
		

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(30, input_dim=self.state_size, activation='relu'))
		model.add(Dense(30, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state):
		self.memory.append((state, action, reward, next_state))#when length crosses 2000 auto deque will occur
		
	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np.random.randint(128,size=(int)(self.action_size/128))
		act_values = self.model.predict(state)
		action = (np.argmax(act_values[0,:128]),np.argmax(act_values[0,128:256]),np.argmax(act_values[0,256:384]),np.argmax(act_values[0,384:]))
		return action  #np.argmax(act_values[0])  # returns action

	def update(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state in minibatch:
			target = self.model.predict(state)
			a = self.model.predict(next_state)
			t = self.model2.predict(next_state)
		   
			t_indices = (np.argmax(a[0,:128]),np.argmax(a[0,128:256]),np.argmax(a[0,256:384]),np.argmax(a[0,384:]))

			#print(target.shape)
			action = np.reshape(action,[1,4])
			target[:,action[:,]] = reward + self.gamma * t[0,t_indices]
			
			self.model.fit(state, target, epochs=1, verbose=0)
		
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
			
		self.counter+=1
		if(self.counter>self.C):
			self.counter = 0
			self.model2.set_weights(self.model.get_weights())
	
	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)


if __name__ == "__main__":
	#env = gym.make('CartPole-v1')
	#state_size = env.observation_space.shape[0]
	#action_size = env.action_space.n
	
	
	#state_size=6,action_size=256*no of controllable objects
	state_size = 6;
	action_size = 128*4;
	
	agent = DQNAgent(state_size, action_size)
	environment = env.Environment()
	# agent.load("./save/cartpole-dqn.h5")
	#done = False
	batch_size = 128
	
	# 1 DAY = 1 EPISODE
	for e in range(EPISODES):
		state = environment.observeState() #observe current state
		state = np.reshape(state, [1, state_size])
		for t in range(96):
			# env.render()
			action = agent.act(state)
			next_state, reward = environment.agentAct(action) #call an environment function with outputted action to get next_state
			#reward = reward if not done else -10
			next_state = np.reshape(next_state, [1, state_size])
			agent.remember(state, action, reward, next_state)
			state = next_state
			'''if done:
				print("episode: {}/{}, score: {}, e: {:.2}"
					  .format(e, EPISODES, time, agent.epsilon))
				break'''
		if len(agent.memory) > batch_size:
			agent.update(batch_size)
		# if e % 10 == 0:
#	 agent.save("./save/cartpole-dqn.h5")
