import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json #also add to agent.py
from keras.models import load_model
from keras.optimizers import Adam
#from TrainedAgent import huber_loss


#import Environment as env
import Agent as ag

agent = ag.DQNAgent(2,128*4)
state = 0


json_file = open('agent_new.json' , 'r')
trained_agent_json = json_file.read()
json_file.close()

trained_agent = model_from_json(trained_agent_json)
trained_agent.load_weights("agent_new.h5")
trained_agent.compile(loss=agent._huber_loss, optimizer=Adam(lr=.001))

'''
trained_agent = load_model('agent_complete.h5') 
print("Agent loaded")
'''
while(True):
    print("Enter state")
    state =  np.asarray([float(x) for x in input().split()])
    state = np.reshape(state,[1,2])
    '''
    if(state == "C"):
        break
	'''
	
    act_values = trained_agent.predict(state)
    action = (np.argmax(act_values[0,:128]),np.argmax(act_values[0,128:256]),np.argmax(act_values[0,256:384]),np.argmax(act_values[0,384:]))
    print(action)


def huber_loss(self, target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
