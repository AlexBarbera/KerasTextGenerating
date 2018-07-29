import keras
from keras.layers import LSTM, Dense, TimeDistributed, Dropout
from keras.models import Sequential

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

def generateText(model):
	pass
