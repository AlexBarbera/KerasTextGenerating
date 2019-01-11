import keras
from keras.layers import LSTM, Dense, TimeDistributed, Dropout
from keras.models import Sequential
import numpy
import sys

def buildModel(vocab, inputs):
	model = Sequential()

	model.add(LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True, stateful=False))

	model.add(LSTM(256, return_sequences=True, stateful=False))
	model.add(Dropout(0.25))

	model.add(LSTM(256, return_sequences=True, stateful=False))
	model.add(Dropout(0.25))

	model.add(LSTM(128, stateful=False))
	model.add(Dropout(0.25))

	#model.add(Dense(80, activation="relu"))
	#model.add(Dropout(0.15))

	model.add(Dense(len(vocab), activation="softmax"))

	model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

	return model

def generateText(model, seed, num, int_to_char, args):
	#print seed[0]
	print "Seed:", "".join([int_to_char[t[0]] for t in seed[0]]) if args.predict_type == "letter" else " ".join([int_to_char[t[0]] for t in seed[0]])
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

