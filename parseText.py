import numpy
from keras.utils import to_categorical

def getInfo(path):
	vocab = []
	char_to_num = []
	num_to_char = []

	text = ""

	with open(path, "r") as f:
		text = f.read()


	vocab = set(list(text))

	del text

	char_to_num = { b:a for a,b in enumerate(vocab) }
	num_to_char = { a:b for a,b in enumerate(vocab) }

	return vocab, char_to_num, num_to_char

def createDataset(path, char_to_int, seq_length=50):
	text = ""
	with open(path, "r") as f:
		text = f.read()
	x = []
	y = []
	for i in xrange(0, len(text) - len(set(text)), 1):
		x.append( [char_to_int[c] for c in text[i:i+seq_length]] ) 
		y.append( char_to_int[text[i+seq_length]] )

	del text

	x = numpy.reshape(x, (len(x), seq_length, 1))
	y = to_categorical(y)

	print x.shape, y.shape

	return x, y
