import random
import re

import string
from typing import Iterable, OrderedDict, Tuple
import unicodedata

import torch
from torch.utils.data import DataLoader, Dataset

from faker import Faker

LETTERS = string.ascii_letters + "'-"
charset_size = len(LETTERS)
char_index = {char: i for i, char in enumerate(LETTERS)}
inverse_index = {i: char for char, i in char_index.items()}
inverse_index[charset_size - 1] = '<EOS>'

TensorPair = Tuple[torch.LongTensor, torch.LongTensor]

def tensorize(word: str) -> TensorPair:
    input_tensor = torch.LongTensor([char_index[char] for char in word])
    eos = (torch.zeros(1) + (charset_size)).type(torch.LongTensor) 
    
    # target_tensor = torch.cat((letter_indices[1:], eos))
    target = torch.cat((input_tensor[1:],eos))
    return input_tensor, target
    
class FrankenDataset(torch.utils.data.Dataset):
    def __init__(self, path:str = 'datasets/data/frankenstein/84-0.txt'):
        with open(path, "r") as f:
            text = f.read()
            text = text.replace("\n", " ").strip()   # remove newlines
            text = re.sub(r" {2,}", " ", text)
            pattern = [
            '[^\x00-\x7f]', # insure file is acii
            '[0-9]',        # remove digits from text
            '[^\w\s]',      # remove punctuation 
            '(_)'           # remove underscores
            ]
            text = re.sub('|'.join(pattern), '', text)
            self.words = text.split(" ")
        
    def __getitem__(self, index:int) -> TensorPair:
        return tensorize(self.words[index])
    
    def __len__(self):
            return len(self.words)

class FakerNameDataset(torch.utils.data.IterableDataset):
    def __init__(self, sample_size = 500):
        self.sample_size = sample_size
        self.namegen = Faker(
            OrderedDict([
                ('en', 5),
                ('fi_FI', 1),
                ('es_ES', 1),
                ('de_DE', 1),
                ('no_NO', 1)
            ])
        )
        
    def __iter__(self) -> Iterable[TensorPair]:
        return (tensorize(self.generate_name()) for _ in range(self.sample_size))
  
    def generate_name(self) -> str:
        name = self.namegen.name()
        name = ''.join(
            char for char in unicodedata.normalize('NFKD', name) if not unicodedata.combining(char)
        )
        name = re.sub("[^a-zA-Z\-' ]", "", name)
           
        if random.uniform(0, 1) < 0.5:
            return name.replace(' ', '')
        else:
            return name.replace(' ', '-')

class CharLSTM(torch.nn.Module):
    def __init__(
        self,
        charset_size: int,
        hidden_size: int,
        embedding_dim: int = 8,
        num_layers: int = 2
    ):
        
        super(CharLSTM, self).__init__()
        
        self.charset_size = charset_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.embedding = torch.nn.Embedding(self.charset_size, self.embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size  = self.charset_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers  = self.num_layers,
            dropout     = 0.5
        )
        
        # self.lstm = nn.LSTM(input_size = self.lstm_size, hidden_size = self.lstm_size, num_layers = self.num_layers, dropout = 0.2, batch_first = False).to(self.device)
        
        self.decoder = torch.nn.Linear(self.hidden_size, self.charset_size)
        self.dropout = torch.nn.Dropout(p = 0.25)
        self.softmax = torch.nn.Softmax(dim = 2)
        
    def forward(self, input_tensor, hidden_state):
        embedded = self.embedding(input_tensor)
        output, hidden_state = self.lstm(input_tensor, hidden_state)
        output = self.decoder(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden_state
    
    def sample(
        self,
        seed: str,
        max_len: int = 8,
        break_on_eos: bool = True,
        eval_mode: bool = True
    ) -> str:
        if eval_mode:
            pass
            # model.eval()
        
    def init_hidden(self):
        return(
            torch.zeros(self.num_layers, 1, self.hidden_size),
            torch.zeros(self.num_layers, 1, self.hidden_size)
        )

        
if __name__ == "__main__":
    frank = FrankenDataset()
    # TODO: Need to understand why and how batch_size changes tensor deminsion(s), currently cannot batch over size of 1 
    dataloader = DataLoader(frank, batch_size=1, shuffle=True)
    for idx, (x,y) in enumerate(dataloader):
        print(f'idx:{idx} - x:{x} - y:{y}')
    
    # fakenames = FakerNameDataset(sample_size=100)
    # fakename_loader = torch.utils.data.DataLoader(fakenames)
    # for idx, (x, y) in enumerate(fakename_loader):
    #     print(f'idx: {idx}, x: {x}, y: {y}')
    
    model = CharLSTM(charset_size, hidden_size=128, embedding_dim=8, num_layers=2)
    print(model)
    print(type(model))