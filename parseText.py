import numpy
from keras.utils import to_categorical
import re

def getVocab(args, full=False):
	text = open(args.path, "r").read()

	vocab = None
	
	if args.predict_type == "letter":
		vocab = set(list(text))
	elif args.predict_type == "word":
		sym = ".,:;()[]\"?!\n"
		asym = [".",",","\"","(",")","?","!","..."]

		for s in sym:
			text = text.replace(s, " ")

		vocab = filter(None, text.split(" "))

		if not full:
			vocab = set(vocab).union(set(asym))
		else:
			vocab += asym
	del text

	return list(vocab)

def getInfo(path, args):
	vocab = []
	char_to_num = []
	num_to_char = []


	vocab = getVocab(args)


	char_to_num = { b:a for a,b in enumerate(vocab) }
	num_to_char = { a:b for a,b in enumerate(vocab) }

	return vocab, char_to_num, num_to_char

def createDataset(path, char_to_int, seq_length, args):
	text = open(args.path, "r").read()

	if args.predict_type == "word":
		text = getVocab(args, full=True)

	x = []
	y = []

	for i in xrange(0, len(text) - seq_length, 1):
		x.append( [char_to_int[c] for c in text[i:i+seq_length]] ) 
		y.append( char_to_int[text[i+seq_length]] )

	del text

	x = numpy.reshape(x, (len(x), seq_length, 1))
	y = to_categorical(y)

	print x.shape, y.shape

	return x, y
