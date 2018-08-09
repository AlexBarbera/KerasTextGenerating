import numpy
from keras.utils import to_categorical
import re

def getVocab(args, to_set=True):
	text = open(args.path, "r").read()

	vocab = None
	
	if args.predict_type == "letter":
		vocab = list(text)
	elif args.predict_type == "word":
		"""
		sym = ".,:;()[]\"?!\n"
		asym = [".",",","\"","(",")","?","!","..."]

		for s in sym:
			text = text.replace(s, " ")

		vocab = filter(None, text.split(" "))
		"""

		vocab = re.split("(\W)", text)

	del text

	if to_set:
		vocab = list(set(vocab))

	return vocab

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
		text = getVocab(args, False)

	x = []
	y = []

	for i in xrange(0, len(text) - seq_length, 1):
		x.append( [char_to_int[c] for c in text[i:i+seq_length]] ) 
		y.append( char_to_int[text[i+seq_length]] )

	x = numpy.reshape(x, (len(x), seq_length, 1))

	if args.normalize_inputs:
		#x /= float(len(text))
		pass

	del text

	y = to_categorical(y)

	print x.shape, y.shape

	return x, y
