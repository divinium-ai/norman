from this import d
from typing_extensions import Self
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import Adam

class GOTLSTM(nn.Module):
    def __init__(self, device, char_to_idx, idx_to_char, hidden_dim, vocab, hidden_size, layers = 1):
        super(GOTLSTM, self).__init__()
        self.device = device  
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.hidden_dim = hidden_dim
        
        self.vocab_size = vocab
        self.hidden_size = hidden_size
        self.layers = layers
        self.lstm = nn.LSTM(vocab, hidden_size, layers, batch_first = False)
        self.linear = nn.Linear(hidden_size, vocab, bias = True)
    
    def forward(self, input, h0=None, c0 = None):
        if h0 == None or c0 == None:
            output, (hn, cn) = self.lstm(input)
        else:
            output, (hn, cn) = self.lstm(input, (h0, c0))
        scores = self.linear(output)
        return scores, hn, cn
    
    def sample(self, x, length = 500):
        x = x.view(1, 1, self.vocab_size)
        h = torch.zeros(self.layers, 1, self.hidden_dim).to(device=self.device)
        c = torch.zeros(self.layers, 1, self.hidden_dim).to(device=self.device)
        
        txt = ''
        for i in range(length):
            scores, h, c = self.forward(x, h, c)
            prob = nn.functional.softmax(scores, dim = 2).view(self.vocab_size)
            predict = torch.tensor(list(WeightedRandomSampler(prob, 1, replacement = True)))
            x = F.one_hot(predict, num_classes = self.vocab_size)
            x = x.view(1, 1, self.vocab_size).type(torch.FloatTensor).to(self.device)
            next_char = self.idx_to_char[predict.item()]
            txt += next_char
        return txt
    
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        combined = torch.cat((combined, hidden), 1)
        output = self.o2o(combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
        