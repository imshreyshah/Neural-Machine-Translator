from typing import List, Dict
import unidecode
import string
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import configparser
config = configparser.ConfigParser()
config.read("dev.config")
config=config["values"]


#Defining the global Tokens.
SOS_TOKEN = 1
EOS_TOKEN = 2
PAD_TOKEN = 0
MAX_LENGTH=20

# This function is preprocessing a single sentence from the database
def preprocess(sentence: str, hindi=False) -> str:
    # remove tabs and newlines
    sentence = ' '.join(sentence.split())
    
    #convert the sentence into lower cases
    sentence = sentence.lower()
    # remove accented chars such as in cafe
    #sentence = unidecode.unidecode(sentence)
    # remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # remove digits
    sentence = sentence.translate(str.maketrans('', '', string.digits))
    # remove hindi digits
    if hindi:
        sentence = re.sub("[२३०८१५७९४६]", "", sentence)
    
    #remove trailing and leading extra white spaces
    sentence = sentence.strip()
    
    #Remove any excess white spaces from within the sentence
    sentence = re.sub(' +', ' ', sentence) 
    
    #append the prepend the SOS token and append the EOS token to the sentence with spaces.
    sentence = "SOS" + " " + sentence + " " + "EOS" 

    ''' for an expected input of sentence = ' hi, my name is  sam  ', output will be 'hi my name is sam'
    '''

    return sentence

#helper function 1. Returns a list of all the unique words in our corpora.
def get_vocab(lang: pd.Series) -> List:
    '''
       For a pd.series object like 
       1. Would you like some tea
       2. do you know my name

       output should be 
       ['do', 'know', 'like', 'my', 'name', 'some', 'tea', 'would', 'you']
    '''
    vocab = []

    for sentence in lang:
        words = sentence.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
    
    vocab.sort()
    
    return vocab


#Helper 2: Creates a dictionary with token-> index mapping. Used in encoding.
def token_idx(words: List) -> Dict:
    
    '''
        input of words ['a','b','c'] -> output should be {'a': 1, 'b' : 2, 'c': 3}
    '''
    idx = {'PAD' : 0, 'SOS' : 1, 'EOS' : 2}
    i = 3
    for word in words:
        if word == "SOS" or word == "EOS":
            continue
        idx[word] = i
        i = i+1
    
    return idx

#Helper 3: Creates a dictionary for index to word mapping. Used in decoding
def idx_token(wor2idx: Dict) -> Dict:
    inv_dict = {v: k for k, v in wor2idx.items()} 
    return inv_dict
 
#Helper 4: Pads sequences to a particular length so that all the sequences are of same length in a batch.
def pad_sequences(sent_tensor: List[List[int]], max_len: int=20) -> np.ndarray:
    padded = np.zeros((max_len), dtype=np.int64)
    if len(sent_tensor) > max_len:
        padded[:] = sent_tensor[:max_len]
    else:
        padded[:len(sent_tensor)] = sent_tensor
    return padded

#Converts a particular sentence to its corresponding numeric tensor using word2idx dictionary.
def convert_to_tensor(word2idx: Dict, sentences: pd.Series):
    tensor = [[word2idx[s] for s in eng.split()]
              for eng in sentences.values.tolist()]
    tensor = [pad_sequences(x) for x in tensor]
    return tensor

#Class of type Dataset. This must contain __len__() and __getitem() functions as a part of their hooks.
class Data(Dataset):
    def __init__(self, input_sent, target_sent):
        self.input_sent = input_sent
        self.target_sent = target_sent

    def __len__(self):
        return len(self.input_sent)

    def __getitem__(self, index):
        x = self.input_sent[index]
        y = self.target_sent[index]
        return x, y

#Main function being called when we need to retrieve inout batch, output batch and DataLoader objects.
def get_dataset(batch_size=2, types="train", shuffle=True, num_workers=1, pin_memory=False, drop_last=True):
    #Read the file 
    lines = pd.read_csv('Hindi_English_Truncated_Corpus.csv', encoding='utf-8')
    #Get lines only with source as "ted"
    lines = lines[lines['source'] == 'ted']  # Remove other sources

    #Remove duplicate lines
    lines.drop_duplicates(subset ="english_sentence", keep = 'first', inplace = True)
    #Random Sample of 25000 sentences
    lines = lines.sample(n=int(config["samples"]), random_state=42)
    
    #Call preprocess functions on all english sentences
    for index, row in lines.iterrows(): 
        lines['english_sentence'][index] = preprocess(lines['english_sentence'][index])
    
    #Call preprocess functions on all hindi sentences
    for index, row in lines.iterrows(): 
        lines['hindi_sentence'][index] = preprocess(lines['hindi_sentence'][index], hindi= True)
    
    #Retrieve length of each english sentence and store it in the lines dataframe under a new column "length_english_sentence"
    lines['length_english_sentence'] = 0
    for index, row in lines.iterrows(): 
        lines['length_english_sentence'][index] = len(lines['english_sentence'][index].split())

    #Retrieve length of each hindi sentences and store it in the lines dataframe under a new column "length_hindi_sentence"
    lines['length_hindi_sentence'] = 0
    for index, row in lines.iterrows(): 
        lines['length_hindi_sentence'][index] = len(lines['hindi_sentence'][index].split())
    
    #Remove all the sentences with lengths greater than equal to max_length.
    for index,row in lines.iterrows():
        if lines['length_english_sentence'][index] >= MAX_LENGTH or lines['length_hindi_sentence'][index] >= MAX_LENGTH:
            lines.drop(index = index,inplace = True)
    
    #Get List of english words      
    english_words = get_vocab(lines['english_sentence'])
    #Get List of Hindi Words
    hindi_words = get_vocab(lines['hindi_sentence'])
    #Get word2idx_eng for english
    word2idx_eng = token_idx(english_words)
    #Get word2idx_hin for hindi
    word2idx_hin = token_idx(hindi_words)
    #get idx2word_eng for english
    idx2word_eng = idx_token(word2idx_eng)
    
    #get idx2word_hin for hindi
    idx2word_hin = idx_token(word2idx_hin)
    
    #Convert the sentences to tensors using dictionaries created above
    #English tensor in input_tensor
    input_tensor = convert_to_tensor(word2idx_eng,lines['english_sentence'])
    
    #Hindi tensor in output_tensor
    target_tensor = convert_to_tensor(word2idx_hin,lines['hindi_sentence'])
    
    #Calling the split function and doing a 80-20 split on the input and target tensors.
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2, random_state=42)

    #Calling the Dataloaders and passing the Dataset type objects created using Data() class.
    if types == "train":
        train_dataset = Data(input_tensor_train, target_tensor_train)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), english_words, hindi_words
    elif types == "val":
        val_dataset = Data(input_tensor_val, target_tensor_val)
        return DataLoader(val_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last), idx2word_eng, idx2word_hin
    else:
        raise ValueError("types must be in ['train','val']")


if __name__ == "__main__":
    get_dataset()
   