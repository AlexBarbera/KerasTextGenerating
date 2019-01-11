# KerasTextGeneration

This repository contains a tool to generate text using RNN (LSTM) using [keras](https://keras.io/ "Keras' Homepage") with [Tensorflow backend](https://www.tensorflow.org/ "Tensorflow's Homepage").
To run an instance using the example data given, you can run:
```
python main.py data/text.txt
```

The repository is structured as follows:
- `main.py` 
  This code is the backbone of the project, calls the rest of the modules to train or generate.

- `modelCreator.py`
   This module creates the RNN and also handles the text generation
- `parseText.py`
   This module is responsible of extracting the vocabulary and the dictionaries of the input text. Also is in charge of generating the dataset for training.

## Parameter options
```
usage: python main.py [-h] [--seq_length SEQ_LENGTH] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--verbose VERBOSE]
               [--gen_length GEN_LENGTH]
               path

Train a RNN to generate text.

positional arguments:
  path                  Path to text file with training text.

optional arguments:
  -h, --help            show this help message and exit
  --seq_length SEQ_LENGTH
                        Sequence length to train.
  --epochs EPOCHS       Number of epochs to train.
  --batch_size BATCH_SIZE
                        Batch size used in training.
  --verbose VERBOSE     Verbosity in training.
  --gen_length GEN_LENGTH
                        Length of generated string.
```
## Data
  - *`text.txt`* Example text from the "Fate Stay Night" series.
  - *`hs.txt`* Example text from the "Fate Stay Night - Heaven's Feel" series.
  - *`ubw.txt`* Example text from the "Fate Stay Night - Unlimited Blade Works" series.
  - *`full.txt`* Example text from all the "Fate Stay Night" series.
  - *`downloadData.py`* Code used to download `text.txt`, can be modified to download  text from any static html page.

## Dependencies
- [keras](https://keras.io/)
- [Tensorflow backend](https://www.tensorflow.org/)
- Numpy

All the dependencies can be fullfilled by running:
`pip install --upgrade keras tensorflow numpy`
