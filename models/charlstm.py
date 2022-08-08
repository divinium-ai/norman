from ast import Num
from calendar import c
from cmath import e
from random import random
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import Adam

class CharLSTM(nn.Module):
    def __init__(self, input_size = 250, hidden_size = 250, output_size = 250):
        super(CharLSTM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size).to(self.device)
        self.i2o = nn.Linear(input_size + hidden_size, output_size).to(self.device)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        combined = torch.cat((combined, hidden), 1)
        output = self.o2o(combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output.to(self.device), hidden.to(self.device)

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(self.device)

class CharLSTM2(nn.Module):
        def __init__(self, vocab, hidden_size, device, dropout=0.1,layers=1):
            super(CharLSTM2, self).__init__()
            self.vocab_size = vocab
            self.hidden_size = hidden_size
            self.device = device
            self.dropout = dropout
            self.layers = layers

            self.lstm = nn.LSTM(self.vocab_size, hidden_size, layers, batch_first=False)
            self.linear = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
            
        def forward(self, x, h0, c0):
            if h0 == None or c0 == None:
                output, (h, c) = self.lstm(x)
            else:
                output, (h, c) = self.lstm(x, (h0, c0))
            scores = self.linear(output)
            return scores, h, c
        
        def sample(self, x, idx_to_char, txt_length = 6):
            x = x.view(1, 1, self.vocab_size)
            h = torch.zeros(self.layers, 1, self.hidden_size).to(self.deive)
            c = torch.zeros(self.layers, 1, self.hidden_size).to(self.device)
            txt = ''
            
            for c in range(txt_length):
                scores, h, c = self.forward(x, h, c)
                prod = nn.functional.softmax(scores, dim=2).view(self.vocab_size)
                pred = torch.tensor(list(WeightedRandomSampler(prod, 1, replacement=1)))
                x = F.one_hot(pred, num_classes=self.vocab_size)
                x = x.view(1, 1, self.vocab_size).type(torch.FloatTensor).to(self.device)
                char = idx_to_char[pred.item]
                txt += char
            return txt
        
        def init_state(self):
            return torch.zeros(1, self.hidden_size).to(self.device)
        
class CharLSTM3(nn.Module):
    def __init__(self, dataset):
        super(CharLSTM3, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Define Embedding Layer
        self.embedding_dim = 250
        self.embedding = nn.Embedding(
            num_embeddings = dataset.vocab_size,
            embedding_dim  = self.embedding_dim).to(self.device)
        
        self.num_layers = 3
        self.lstm_size = 250
        self.lstm = nn.LSTM(
            input_size  = self.lstm_size, 
            hidden_size = self.lstm_size,
            num_layers  = self.num_layers,
            dropout = 0.2,
            batch_first = False).to(self.device)
    
        # Define the final fully connected output
        self.fc = nn.Linear(self.lstm_size, dataset.vocab_size).to(self.device)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x).to(self.device)
        out, state = self.lstm(embed, prev_state)
        logits = self.fc(out)
        return logits, state
    
    def init_state(self, seq_length):
        return (torch.zeros(self.num_layers, seq_length, self.lstm_size).to(self.device), 
                torch.zeros(self.num_layers, seq_length, self.lstm_size).to(self.device))
        