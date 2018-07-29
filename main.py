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

	output.add_argument("--epochs", type=int, help="Number of epochs to train.", default=10)
	output.add_argument("--batch_size", type=int, help="Batch size used in training.", default=64)
	output.add_argument("--verbose", type=int, help="Verbosity in training.", default=1)

	output.add_argument("--gen_length", type=int, help="Length of generated string.", default=500)

	return output.parse_args(args)

if __name__ == "__main__":
	args = getArgs(sys.argv[1:])
	print args

	path = args.path
	seq_length = args.seq_length

	vocab, char_to_num, num_to_char = parseText.getInfo(path)
	x, y = parseText.createDataset(path, char_to_num, seq_length)

	model = modelCreator.buildModel(vocab, x)

	model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, verbose=args.verbose)

	"""numpy.random.random_integers(0, len(x) -1)"""
	modelCreator.generateText(model, numpy.reshape(x[0], (1, x.shape[1], x.shape[2])), args.gen_length, num_to_char)
