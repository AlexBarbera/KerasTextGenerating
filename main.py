import keras
import modelCreator
import parseText
import sys
import argparse
import numpy

from keras.callbacks import ModelCheckpoint

def getArgs(args):
	output = argparse.ArgumentParser(description='Train a RNN to generate text.')

	output.add_argument("path", type=str, help="Path to text file with training text.")
	output.add_argument("--seq_length", type=int, help="Sequence length to train.", default=50)

	output.add_argument("--epochs", type=int, help="Number of epochs to train.", default=10)
	output.add_argument("--batch_size", type=int, help="Batch size used in training.", default=200)
	output.add_argument("--verbose", type=int, help="Verbosity in training.", default=1)

	output.add_argument("--gen_length", type=int, help="Length of generated string.", default=500)

	output.add_argument("--save_path", type=str, help="Directory to save model and weights of training, also used to load.", default=".")

	return output.parse_args(args)

if __name__ == "__main__":
	args = getArgs(sys.argv[1:])
	print args

	path = args.path
	seq_length = args.seq_length

	vocab, char_to_num, num_to_char = parseText.getInfo(path)
	x, y = parseText.createDataset(path, char_to_num, seq_length)

	model = modelCreator.buildModel(vocab, x)

	with open(args.save_path + "/model.json", "w") as f:
		f.write(model.to_json())

	c = [ModelCheckpoint(args.save_path + "/weights_model-{epoch:02d}-{loss:.5f}.hdf5", monitor="loss", verbose=1, save_best_only=True, mode="min")]

	model.fit(x, y, batch_size=args.batch_size, epochs=args.epochs, verbose=args.verbose, callbacks=c)

	"""numpy.random.random_integers(0, len(x) -1)"""
	modelCreator.generateText(model, numpy.reshape(x[0], (1, x.shape[1], x.shape[2])), args.gen_length, num_to_char)
