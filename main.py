######################################################################
# Training and Evaluating
# =======================
from seq2seq import EncoderRNN,Linear,DecoderRNN
from train import trainIters
import configparser
config = configparser.ConfigParser()
config.read("dev.config")
config=config["values"]
from preprocess import get_dataset
import argparse
import torch
import os
SOS_token = 1   # SOS_token: start of sentence
EOS_token = 2   # EOS_token: end of sentence

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")
    # Parser
parser = argparse.ArgumentParser(description='Creating Classifier')
##############
# Cuda Flags #
##############
if config["cuda"]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

###############################
# Creating the dataset object #
###############################
# Create training data object
bidirectional=config.getboolean("bidirectional")
trainset,source_vocab,target_vocab = get_dataset(types="train",batch_size=int(config["batch_size"]),shuffle=True,num_workers=int(config["num_workers"]),pin_memory=False,drop_last=True)
encoder1 = EncoderRNN(int(config["hidden_size_encoder"]),len(source_vocab)+2,int(config["batch_size"]), num_layers=int(config["num_layer_encoder"]),bidirectional=bidirectional).to(device)
bridge = Linear(bidirectional, int(config["hidden_size_encoder"]), int(config["hidden_size_decoder"])).to(device)
decoder1 = DecoderRNN(int(config["hidden_size_decoder"]),len(target_vocab)+2, int(config["batch_size"]), num_layers=int(config["num_layer_decoder"])).to(device)
trainIters(trainset,encoder1, decoder1, bridge,num_epochs=int(config["num_epoch"]),batch_size=int(config["batch_size"]),print_every=10,device=device)
torch.save(encoder1,"models/encoder.pt")
torch.save(decoder1,"models/decoder.pt")
torch.save(bridge,"models/bridge.pt")
