from __future__ import print_function
import keras
from keras.layers import LSTM, Dense, TimeDistributed, Dropout
from keras.models import Sequential
import numpy

def buildModel(vocab, inputs):
	model = Sequential()

	model.add(LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))

	model.add(LSTM(256, return_sequences=True))
	model.add(Dropout(0.35))

	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.45))

	model.add(LSTM(100))
	model.add(Dropout(0.25))

	model.add(Dense(80, activation="relu"))
	model.add(Dropout(0.15))

	model.add(Dense(len(vocab), activation="softmax"))

	model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

	return model

def generateText(model, seed, num, int_to_char):
	x = seed

	for i in xrange(num):
		res = model.predict(x)

		for i in xrange(x.shape[1]-1):
			x[0][i][0] = x[0][i+1][0]

		x[0][-1][0] = numpy.argmax(res)

		res = int_to_char[numpy.argmax(x)]
		
		print(res, end="", flush=True)

	print("\n\n")

