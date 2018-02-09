import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json #also add to agent.py
#from keras.optimizers import Adam

import Environment as env
import Agent as ag


#agent = ag.DQNAgent(6,128*4)

json_file = open('agent.json' , 'r')
trained_agent_json = json_file.read()
json_file.close()

trained_agent = model_from_json()
trained_agent.load_weights("agent.h5")
trained_agent.compile(loss='mse', optimizer=Adam(lr=.001)

while True:
                      print("Enter state")
                      state = input()
                      if(state == "C"):
                        break
                      act_values = trained_agent.predict(state)
                      action = (np.argmax(act_values[0,:128]),np.argmax(act_values[0,128:256]),np.argmax(act_values[0,256:384]),np.argmax(act_values[0,384:]))
                      print(action)
                      
  
