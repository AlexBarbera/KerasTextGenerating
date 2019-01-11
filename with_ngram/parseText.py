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
		y.append( [char_to_int[temp] for temp in text[i+seq_length:i+seq_length + args.n] ] )


	y = y[:-args.n - 1]#los ultimos n-1 elementos no son ngramas enteros
	x = numpy.reshape(x, (len(x), seq_length, 1))

	x = x[:-args.n-1]#drop last n-1 por la misma razon

	if args.normalize_inputs:
		#x = x / float(len(char_to_int))
		x = (x - numpy.mean(x)) / numpy.std(x)

	del text

	#print y

	y = to_categorical(y)
	y = numpy.reshape( y, (len(y), args.n * len(char_to_int)) )

	print x.shape, y.shape


	return x, y
