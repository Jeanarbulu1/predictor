import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from read_data import generator
import pickle
try:	
	model = Sequential()
	model.add(LSTM(32, input_shape= (2,1)))
	model.add(Dense(2,kernel_initializer='normal',activation='linear'))
	model.add(Dense(1,kernel_initializer='normal',activation='linear'))
	model.compile(loss='mse',optimizer ='adam',metrics=['accuracy'])
	model.fit_generator(generator(batch_size = 20), samples_per_epoch=50, nb_epoch=10)
	filename = 'speech_predictor.sav'
	pickle.dump(model, open(filename, 'wb'))
except MemoryError:
	print(MemoryError)