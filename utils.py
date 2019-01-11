import argparse


def getArgs(args):
	output = argparse.ArgumentParser(description='Train a RNN to generate text.')

	output.add_argument("path", type=str, help="Path to text file with training text.")
	output.add_argument("--seq-length", type=int, help="Sequence length to train.", default=50)
	output.add_argument("--predict-type", choices=["letter", "word"], default="letter")

	output.add_argument("--epochs", type=int, help="Number of epochs to train.", default=10)
	output.add_argument("--batch-size", type=int, help="Batch size used in training.", default=200)
	output.add_argument("--verbose", type=int, help="Verbosity in training.", default=1)
	output.add_argument("--random", action="store_true", help="Randomize training batches.")

	output.add_argument("--gen-length", type=int, help="Length of generated string.", default=500)

	output.add_argument("--save-path", type=str, help="Directory to save model and weights of training, also used to load.", default=".")

	output.add_argument("--normalize-inputs", action="store_true", help="To normalize network inputs.")

	output.add_argument("--model-type", type=int, help="Type of model to use (0. LSTM, 1.CNN, 2:Bi-LSTM).", default=1)

	return output.parse_args(args)
