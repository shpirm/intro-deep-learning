# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings

   Hande Celikkanat

   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import spacy
import regex as re
import time
from tqdm import tqdm
from torchtext import vocab
from pathlib import Path

import data

# Constants - Add here as you wish
N_EPOCHS = 5
EMBEDDING_DIM = 200
N_CLASSES = 2
LEARNING_RATE = 0.001

FILE = Path(__file__).parent.parent
TRAIN_FILE = str(FILE) + '/data/sent140.train.midi.csv'
DEV_FILE   = str(FILE) + '/data/sent140.dev.csv'
TEST_FILE  = str(FILE) + '/data/sent140.test.csv'

TRAIN_BS = 5000
DEV_BS   = 100
TEST_BS  = 100

# Auxilary functions for data preparation
tok = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in tok(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()


# Evaluation functions
def evaluate(model, loader, criterion):    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs, lengths, labels = batch['inputs'], batch['lengths'], batch['labels']
            lengths, perm_index = lengths.sort(0, descending = True)

            inputs = inputs[:, perm_index]
            labels = labels[perm_index].long()

            output = model(inputs, lengths)

            output = model(inputs, lengths)
            loss = criterion(output, labels)

            output = torch.argmax(output, 1)
            epoch_acc += torch.sum(labels == output) / len(labels)
            epoch_loss += loss

    return epoch_loss / len(loader), epoch_acc / len(loader)


# Utility
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Recurrent Network
class RNN(nn.Module):
    def __init__(self, hidden_dim = 64):
        super().__init__()
        
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings)

        self.rnn = nn.RNN(EMBEDDING_DIM, hidden_dim, batch_first=False)   
        self.fc = nn.Linear(hidden_dim, N_CLASSES)


    def forward(self, inputs, lengths):
        embedding = self.embedding(inputs) 
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedding, lengths)

        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        output = self.fc(hidden.squeeze(0))
        return F.log_softmax(output, 1)
    

if __name__ == '__main__':
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print('Prepariong dataset')
        train_loader, dev_loader, test_loader, glove_embeddings = data.get_dataset(
                tokenizer,
                TRAIN_FILE,
                DEV_FILE,
                TEST_FILE,
                TRAIN_BS,
                DEV_BS,
                TEST_BS,
                EMBEDDING_DIM) 

        model = RNN()

        criterion = torch.nn.NLLLoss()   
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 

        # --- Train Loop ---
        print('Training')
        for epoch in range(N_EPOCHS):
            print(f'Epoch {epoch}')
            start_time = time.time()
            epoch_loss = 0
            epoch_acc = 0
            
            for batch in train_loader:
                inputs, lengths, labels = batch['inputs'], batch['lengths'], batch['labels']
                lengths, perm_index = lengths.sort(0, descending = True)

                inputs = inputs[:, perm_index]
                labels = labels[perm_index].long()

                output = model(inputs, lengths)
                optimizer.zero_grad()
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                output = torch.argmax(output, 1)
                epoch_acc += torch.sum(labels == output) / len(labels)
                epoch_loss += loss

                train_loss, train_acc = (epoch_loss / len(train_loader), epoch_acc / len(train_loader)) 
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
 
            train_loss, train_acc = (epoch_loss / len(train_loader), epoch_acc / len(train_loader)) 
            valid_loss, valid_acc = evaluate(model, dev_loader, criterion)
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

