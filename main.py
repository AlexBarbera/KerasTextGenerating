import keras
import modelCreator
import parseText
import sys
import argparse
import numpy

def getArgs(args):
	output = argparse.ArgumentParser(description='Train a RNN to generate text.')

	output.add_argument("path", type=str, help="Path to text file with training text.")
	output.add_argument("--seq_length", type=int, help="Sequence length to train.", default=50)

	return output.parse_args(args)

if __name__ == "__main__":
	args = getArgs(sys.argv[1:])
	print args

	path = args.path
	seq_length = args.seq_length

	vocab, char_to_num, num_to_char = parseText.getInfo(path)
	x, y = parseText.createDataset(path, char_to_num, seq_length)

	model = modelCreator.buildModel(vocab, x)

	modelCreator.generateText(model, numpy.reshape(x[0], (1, x.shape[1], x.shape[2])), 50, num_to_char)

	#model.fit(x, y, batch_size=64, epochs=10, verbose=1)
