import keras
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, BatchNormalization, Bidirectional
from keras.layers import Input, Embedding, Conv1D, Flatten
from keras.models import Sequential
import numpy
import sys

def buildModel(vocab, inputs):
	model = Sequential()

	model.add(LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True, stateful=False))

	model.add(LSTM(256, return_sequences=True, stateful=False))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(LSTM(256, return_sequences=True, stateful=False))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(LSTM(128, stateful=False))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	#model.add(Dense(80, activation="relu"))
	#model.add(Dropout(0.15))

	# model.add(Dense(2*len(vocab), activation="relu"))
	model.add(Dense(len(vocab), activation="softmax"))

	model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def buildModelCNN(vocab, inputs):
	model = Sequential()
	start = Input(shape=(inputs.shape[1], inputs.shape[2]))

	#end = Embedding(len(vocab), 1028)(start)
	
	end = Conv1D(filters=128, kernel_size=4, activation="relu")(start)
	end = Conv1D(filters=64, kernel_size=4, activation="relu")(end)
	end = Conv1D(filters=32, kernel_size=4, activation="relu")(end)
	end = Dropout(0.2)(end)

	end = Flatten()(end)

	end = Dense(len(vocab), activation="softmax")(end)
	
	model = keras.models.Model(inputs=start, outputs=end)

	model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def buildModelBiDir(vocab, inputs):
	model = Sequential()

	model.add(Bidirectional(LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True, stateful=False)))

	model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Bidirectional(LSTM(256, return_sequences=True, stateful=False)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Bidirectional(LSTM(128, stateful=False)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))

	#model.add(Dense(80, activation="relu"))
	#model.add(Dropout(0.15))

	model.add(Dense(2*len(vocab), activation="relu"))
	model.add(Dense(len(vocab), activation="softmax"))

	model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def generateText(model, seed, num, int_to_char, args):
	#print seed[0]
	print "Seed:", "".join([int_to_char[t[0]] for t in seed[0]]) if args.predict_type == "letter" else " ".join([int_to_char[t[0]] for t in seed[0]])
	print "--------------------------------\n\n"
	x = seed[0]
	for i in xrange(num):
		x = numpy.reshape(x, (1, len(x), 1))
		res = model.predict(x)

		m = sorted(res[0], reverse=True)[3]
		
		mask = numpy.where(res[0] >= m, 1, 0)
		#print m, mask

		res[0] *= mask
		index = numpy.random.choice(len(res[0]), 1, p=res[0]/numpy.sum(res))[0]
		#index = numpy.where(numpy.random.multinomial(1, res[0]/numpy.sum(res)) == 1)[0][0]
		#print "Seed:", "".join([int_to_char[t[0]] for t in x[0]])
		sys.stdout.write(int_to_char[index])

		if args.predict_type == "word":
			sys.stdout.write(" ")
		sys.stdout.flush()

		x = [t[0] for t in x[0]]
		x.append(index)
		x = numpy.asarray(x[1:])

	print ""

		
