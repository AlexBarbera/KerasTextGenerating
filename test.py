import sys
import modelCreator
import parseText
import numpy
import utils
from keras.models import model_from_json
import argparse

if __name__ == "__main__":
	args = utils.getArgs(sys.argv[1:])
	vocab, char_to_num, num_to_char = parseText.getInfo(args.path, args)
	x, y = parseText.createDataset(args.path, char_to_num, args.seq_length, args)
	model = None

	with open(args.save_path + "/model.json", "r") as f:
		model = model_from_json(f.read())

	model.load_weights(args.save_path + "/weights.hdf5")

	modelCreator.generateText(model,numpy.reshape(x[0], (1, x.shape[1], x.shape[2])), 200, num_to_char, args)
