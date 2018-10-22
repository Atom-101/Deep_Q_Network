from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json

model = Sequential()
model.add(Dense(24, input_dim=2, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(512, activation='linear'))

model_json = model.to_json()
with open ("agent.json","w") as json_file:
    json_file.write(model_json)
