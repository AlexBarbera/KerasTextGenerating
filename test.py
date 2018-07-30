import sys
import modelCreator
import parseText
import numpy
from keras.models import model_from_json

if __name__ == "__main__":
	vocab, char_to_num, num_to_char = parseText.getInfo(sys.argv[1])
	x, y = parseText.createDataset(sys.argv[1], char_to_num, 50)
	model = None

	with open(sys.argv[3], "r") as f:
		model = model_from_json(f.read())

	model.load_weights(sys.argv[2])

	modelCreator.generateText(model, x[0], 200, num_to_char)
