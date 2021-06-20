# Neural Machine Translator from English to Hindi
This project consists of a Seq2Seq Neural Machine Translator on English-Hindi data. The architecture is a simple encoder and decoder with uni-directional LSTM and feed-forward network. THis project was done as a part of the course: - Neural Networks and Fuzzy Logic at BITS Pilani. 

## Code Files

[main.py](main.py): Contains all code to instantiate appropriate classes and call appropriate functions in
the correct order from all other files.

[validate.py](validate.py): Contains code to test the model once it has been trained.

[trainer.py](trainer.py): Contains code to train the model.

[seq2seq.py](seq2seq.py): Contains the architecture of the model.

[preprocess.py](preprocess.py): Code to load and pre-process the dataset.

Utils/ : some extra utility functions.

## Dataset

We are using a English-Hindi translated corpus, and it can be downloaded from [here](https://drive.google.com/file/d/1BI5piDKMaZHvmJ6UoHidVyrxI1NmKilP/view?usp=sharing). Download the CSV file and keep it in the root directory. 

## How to Run

Optionally, create a virtual environment on your system and open it. 

To run the project, first clone the repository by typing the command in git bash.
```
git clone https://github.com/imshreyshah/Neural-Machine-Translator.git
```

Alternatively, you can download the code as .zip and extract the files.

Shift to the cloned directory
```
cd Neural-Machine-Translator
```

To install the requirements, run the following command:
```
pip install -r requirements.txt
```

To create our architecture and train it, run the following command: 
```
python main.py
```

The models will be saved in the ```models``` directory. 

Run the validation script using the following command: 
```
python validate.py
```

To test the seq2seq model, run the following command: 
```
python test_seq2seq.py
```

## Acknowledgement
Thanks to the Instructor and the Teacher's Assistants for implementing many functionalities of this assignment.
