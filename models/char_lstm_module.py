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
    
class char_lstm(nn.Module):
    def __init__(self, dataset_vocab_length, lstm_size = 128, embeddings = 128, layers = 5):
        super(char_lstm, self).__init__()
        # self.vocab_length = vocab_length
        self.lstm_size = lstm_size
        self.layers = layers
        self.embedding_dim = embeddings

        self.embedding = nn.Embedding(dataset_vocab_length, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size  = self.lstm_size, 
            hidden_size = self.lstm_size,
            num_layers  = self.layers,
            dropout = 0.2,
        )
        
        self.fc = nn.Linear(self.lstm_size, dataset_vocab_length)
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
    
    # https://learning.oreilly.com/library/view/pytorch-recipes-a/9781484242582/html/474315_1_En_7_Chapter.xhtml
    def initHidden(self):
        return torch.zeros(1, self.lstm_size)