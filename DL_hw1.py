"""
   Introduction to Deep Learning (LDA-T3114)
   Skeleton Code for Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Hande Celikkanat & Miikka Silfverberg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

#ATTENTION: If necessary, add the paths to your data_semeval.py and paths.py here:
#import sys
#sys.path.append('</path/to/below/modules>')
from data_semeval import *
from paths import data_dir


#--- hyperparameters ---

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 15
LEARNING_RATE = 1
BATCH_SIZE = 50
REPORT_EVERY = 1
IS_VERBOSE = False


def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])



#--- model ---

class FFNN(nn.Module):
    # Feel free to add whichever arguments you like here.
    def __init__(self, vocab_size, n_classes, extra_arg_1 = 8, extra_arg_2 = 8):
        super(FFNN, self).__init__()
        # WRITE CODE HERE
        self.input = nn.Linear(vocab_size, extra_arg_1)
        self.hidden = nn.Linear(extra_arg_1, extra_arg_2)
        self.output = nn.Linear(extra_arg_2, n_classes)

    def forward(self, x):
        # WRITE CODE HERE
        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x



#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)



#--- set up ---

# WRITE CODE HERE
model = FFNN(vocab_size, N_CLASSES) #add extra arguments here if you use
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)



#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])    

    for i in range(int(len(data['training'])/BATCH_SIZE)):
        minibatch = data['training'][i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        # WRITE CODE HERE 
        batch_x = torch.cat([tweet['BOW'] for tweet in minibatch], 0)
        batch_y = torch.cat([label_to_idx(tweet['SENTIMENT']) for tweet in minibatch], 0)
    
        output = model.forward(batch_x)
        loss = loss_function(output, batch_y)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                              
    if ((epoch+1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch+1, total_loss*BATCH_SIZE/len(data['training'])))



#--- test ---
correct = 0
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])

        # WRITE CODE HERE
        # You can, but for the sake of this homework do not have to,
        # use batching for the test data.
        predicted = model.forward(torch.Tensor(tweet['BOW']))
        predicted = torch.argmax(predicted)

        if predicted == gold_class: correct += 1

        if IS_VERBOSE:
            print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))

