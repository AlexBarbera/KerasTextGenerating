import keras
import model as modelCreator
import parseText

if __name__ == "__main__":
	path = "data/text.txt"
	seq_length = 50

	vocab, char_to_num, num_to_char = parseText.getInfo(path)
	x, y = parseText.createDataset(path, char_to_num, seq_length)

	model = modelCreator.buildModel(vocab, x)

	model.fit(x, y, batch_size=64, epochs=10, verbose=1)
