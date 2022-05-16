# Novel Generator
Generative transformer model.
Learns natural language from text corpus.

## Tokenization
Training data and generated text encoded using ByteLevelBPETokenizer implemented by [HuggingFace](https://github.com/huggingface/tokenizers).
Token dictionary needs to be generated beforehand from training, evaluation and test datasets. Resulting dictionary should be saved into JSON file.

## Running
Currently only way to setup the process is by editing main.py scenario.
There are a number of parameters at the beginning of the file which should be set.
These include path to dictionary JSON-file, paths to files with corpus data for training, etc...
There are also hyperparameters used to create the model and train and infer it.
