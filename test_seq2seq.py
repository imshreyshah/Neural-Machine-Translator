## Put this file in the SAME DIRECTORY as seq2seq.py to run these tests

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os

import torch
import torch.utils.data
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from preprocess import get_dataset
from utils.transformer import *
import argparse
from seq2seq import EncoderRNN, DecoderRNN , Linear

device = 'cpu'
#***********************
#*******IMPORTANT*******
#***********************
#PLEASE NOTE THE VALUE  = 2 or 1 for input_size GIVEN BELOW IS JUST FOR TESTING PURPOSES. PLEASE DO NOT HARDCODE ANY VALUES IN YOUR seq2seq.py file

torch.manual_seed(0)
encoder  = EncoderRNN(hidden_size = 1, input_size = 1, batch_size = 1)
decoder  = DecoderRNN(hidden_size = 1, output_size = 2, batch_size = 1)
dense = Linear(bidirectional  = False, hidden_size_encoder = 1 , hidden_size_decoder = 1)
dense2 = Linear(bidirectional  = False, hidden_size_encoder = 2 , hidden_size_decoder = 1)

layers = [encoder, decoder , dense]

layers_individual = []


for layer in layers: 
	for name,module in layer.named_modules():
		if(name==''):
			continue
		else:
 			layers_individual.append([name,module])
  
a = torch.zeros((1)).to(device).long()
b = encoder.initHidden(device=device)

out , hid = encoder(a,b)
print(out)

dec_out,dec_hin = decoder(torch.tensor([1],device=device),hid)
print(dec_out)

lin_out1 = dense(torch.ones(1))
print(lin_out1)

lin_out2 = dense2(torch.ones((3,2)))
print(lin_out2)


"""
Expected output of above statement:
tensor([[[-0.1330]]], grad_fn=<StackBackward>)
tensor([[ 0.1621, -0.8648]], grad_fn=<AddmmBackward>)
tensor([1.])
tensor([[0.5471],
        [0.5471],
        [0.5471]], grad_fn=<AddmmBackward>)

"""
