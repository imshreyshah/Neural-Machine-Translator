######################
# Training the Model #
######################
from preprocess import SOS_TOKEN
from torch import optim
import torch
import torch.nn as nn
from utils.transformer import *
from torch import device

#The train function is called for each batch for each epoch
#An epoch is completed once we iterate through all the examples in out training data
def train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, encoder_optimizer, decoder_optimizer, bridge_optimizer,device, criterion, max_length,batch_size=32,bidirectional=False,teacher_forcing=True):
    # To train, each element of the input sentence will be fed to the encoder.
    # At the decoding phase``<SOS>`` will be fed as the first input to the decoder
    # and the last hidden (state,cell) of the encoder will play the role of the first hidden (cell,state) of the decoder.

    """
    It is very important to initialize the gradients to zero. 
    We want the gradients to be emptied before each batch. Remember in sgd 
    we update the weights after each example and here we update after each batch
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    bridge_optimizer.zero_grad()

    # Define a list for the last hidden layer
    encoder_hiddens_last = []

    #Initializing the loss to zero
    loss = 0

    
    for step_idx in range(batch_size):
        
        """
        Encoder is an object of class EncoderRNN. 
        """
        #Call the method initHidden of encoder class
        encoder_hidden = encoder.initHidden(device) 

        """
        We will take the particular batch input and will select only those values which are non-zero.
        So select the input tensor value for this particular batch and select only the non zero values
        """

        #Selecting the non-zero values from the step tensor after initialising it
        input_tensor_step = input_tensor[:, step_idx][input_tensor[:, step_idx] != 0]

        #After filtering out non-zero values, the size must be less than or equal to max_length.
        input_length = input_tensor_step.size(0)

        

        #initializing output tensor for encoder output with size (batch_size, max_length, encoder.hidden_size) and assign it to appropriate device
        #In this case it must be (32,20,256)
        encoder_outputs = torch.zeros(batch_size,max_length,encoder.hidden_size) 

        #Iterating over each input word individually
        for ei in range(input_length):

            #Call the encoder with input as the input tensor step at index ei and hidden as the hidden state of last stage
            encoder_output, encoder_hidden = encoder(input_tensor_step[ei],encoder_hidden) 
            
            #Store encoder output at each word at each batch in the encoder outputs tensor
            encoder_outputs[step_idx, ei, :] = encoder_output[0, 0]

        # only return the hidden and cell states for the last layer and pass it to the decoder
        #Assign the last encoder hidden states
        hn, cn = encoder_hidden 

        #Reshaping and selecting the last hidden states.
        encoder_hn_last_layer = hn[-1].view(1,1,-1)
        encoder_cn_last_layer = cn[-1].view(1,1,-1)

        #Storing the  last hidden states in a list for further use.
        encoder_hidden = [encoder_hn_last_layer, encoder_cn_last_layer]

        # A linear layer to establish the connection between the encoder/decoder layers.
        encoder_hidden = [bridge(item) for item in encoder_hidden]

        #Appending the hidden states for each layer to the last hidden states.
        encoder_hiddens_last.append(encoder_hidden)

    #################
    #### DECODER ####
    #################

    #Assign the decoder input as a torch tensor with value [SOS]
    decoder_input = torch.tensor([SOS_TOKEN],device = device)

    #Assign the initial decoder hidden states as the last hidden states of encoder retrieved above
    decoder_hiddens = encoder_hiddens_last 

    # teacher_forcing uses the real target outputs as the next input
    # rather than using the decoder's prediction.
    #You can read more about teacher forcing online
    if teacher_forcing:

        for step_idx in range(batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence
            
            #Retrieve the step_idx for the target tensor.
            target_tensor_step = target_tensor[:, step_idx]

            #Select all the non zero values to reduce computation
            target_tensor_step=target_tensor_step[target_tensor[:, step_idx] != 0]

            #Truncated length after filtering out non-zero values.
            target_length = target_tensor_step.size(0)

            #Assigning initial decoder hidden stage. Select the correct indexed encoder stage from the decoder_hiddens tensor.
            decoder_hidden = decoder_hiddens[step_idx]
            # Teacher forcing: Feed the target as the next input

            #Iterating over each word in target
            for di in range(target_length):

                #Call the decoder object and pass the decoder input and hidden state as the hidden state
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden) 
                
                #Add loss for each decoder output and target_tensor for each word in target language
                loss += criterion(decoder_output, target_tensor_step[di].view(1))

                #This is the key step. The next decoder input must be taken from target_tensor_step which is the groun truth.
                #Assign the tensor output step at index di(remember it is 1 step behind)
                decoder_input =  target_tensor_step[di]# Teacher forcing
       
        #find loss normalised with batch size
        loss = loss/batch_size 

    else:
        for step_idx in range(batch_size):
            # reset the LSTM hidden state. Must be done before you run a new sequence. Otherwise the LSTM will treat
            # the new input sequence as a continuation of the previous sequence

            target_tensor_step = target_tensor[:, step_idx][target_tensor[:, step_idx] != 0]
            target_length = target_tensor_step.size(0)
            decoder_hidden = decoder_hiddens[step_idx]

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor_step[di].view(1))
                if decoder_input.item() == EOS_token:
                    break
        loss = loss/batch_size 

    #Call Backward prop for loss
   
    loss.backward()
    #Call Weight update for optimizers. Note that we are doing this after each batch is passed
    #The bridge layer is not being used since we dont have a bidirectional network and hence no weight updation is needed.
    
    encoder_optimizer.step() 
    decoder_optimizer.step()

    #Return loss value normalised with target length
    return loss.item()/target_length 


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

#Converts the time in seconds to minutes
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#Returns the time elapsed
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




def trainIters(trainloader,encoder, decoder, bridge,device,bidirectional=False,teacher_forcing=True,num_epochs=600,batch_size=32,max_length=20, print_every=1000, plot_every=100, learning_rate=0.1):
    #Start the time for keeping track of epoch time
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    #Optimizers being defined for the layers for weight updation. Stochastic Gradient being used here as the optimizer.
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    bridge_optimizer = optim.SGD(bridge.parameters(), lr=learning_rate)

    #Defining the Loss function for our training 
    criterion = nn.CrossEntropyLoss()
    
    #Defines number of times we will be iterating in one epoch
    n_iters_per_epoch = int(len(trainloader) /batch_size)
    
    #Training starts here
    for i in range(num_epochs):
        #Just random print functions.
        print("epoch {} begins".format(i))
        
        for iteration,data in enumerate(trainloader, 1):

            #Assign the data to training_pair
            training_pair = data 

            # Assign the input tensor 
            input_tensor = training_pair[0] 

            #This will process the tensor and returns in required format
            input_tensor, mask_input = reformat_tensor_mask(input_tensor)

            # Assign the target tensor
            target_tensor = training_pair[1] 
            target_tensor, mask_target = reformat_tensor_mask(target_tensor)
            
            #Moving the tensors to gpus for faster calculations
            if device == torch.device("cuda"):
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
            
            #The train function returns the loss for each batch
            #Call the train function and fill in the arguments from above
            loss = train(input_tensor, target_tensor, mask_input, mask_target, encoder, decoder, bridge, encoder_optimizer, decoder_optimizer, 
            bridge_optimizer,device, criterion, max_length,batch_size,bidirectional=False,teacher_forcing=True)
            
            #Printing losses every few iterations
            print_loss_total += loss
            plot_loss_total += loss

            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iteration / n_iters_per_epoch),
                                             iteration, iteration / n_iters_per_epoch * 100, print_loss_avg))

            if iteration % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0


        print('####### Finished epoch %d of %d ########' % (i+1,num_epochs))
