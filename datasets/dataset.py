import re
import string
from logging import raiseExceptions

import numpy as np
import pandas as pd 
from gensim.parsing.preprocessing import remove_stopwords

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from keras.utils import np_utils

from data.nasdaq import nasdaq

sources = {
    'drug':'datasets/data/Drugnames/drugnames-20220512-parquet.gzip',
    'drug_txt': 'datasets/data/Drugnames/drugname.txt',
    'characters': 'datasets/data/marvels/characters-20220517-parquet.gzip',
    'got': 'datasets/data/got/gameofthrones.txt',
    'medical': 'datasets/data/medical/wordlist.txt'
}

class Nasdaq(Dataset):
    def __init__(self, seq_length:int = 8,):
        # Basic Running params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Text Setup
        self.txt = self.load()
        self.charset = sorted(list(set(self.txt)))
        self.seq_length = seq_length
        
        # Encoding
        self.idx_to_char = {idx: char for idx, char in enumerate(self.txt)}
        self.char_to_idx = {char: idx for char, idx in self.idx_to_char.items()}
        self.encoding = [self.char_to_idx[char] for char in self.txt]  
          
    def __getitem__(self, idx):
        x = torch.tensor(self.encoding[idx:idx + self.seq_length]).to(self.device)
        y = torch.tensor(self.encoding[idx + 1:idx + self.seq_length + 1]).to(self.device)
        return (x, y)
    
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def load(self) -> str:
        return nasdaq.getNasdaq()['Company Name']
        
class DomainNames(Dataset):
    def __init__(self, topLevel:str = '.ai') -> None:
        pass
    def __getitem__(self, index):
        pass
    def __len__(self) -> int:
        pass
class Frankenstein(Dataset):
    def __init__(self, seq_length = 8):
        self.charset = string.ascii_letters + "-'"
        self.charset_size = len(self.charset)
        self.seq_length = seq_length
        
        self.char_index = {char: i for i, char in enumerate(self.charset)}
        inverse_index = {i: char for char, i in self.char_index.items()}
        self.inverse_index = inverse_index[self.charset_size -1]
    
    def __len__(self):
        return self.charset_size - self.seq_length
    
    def __getitem__(self, index):
        pass
    
    def tensorize(self, word:str):
        input_tensor = torch.LongTensor([self.char_index[char] for char in word])
        eos = torch.zeros(1).type(torch.LongTensor) + (self.charset_size - 1)
        

class drug(Dataset):
    def __init__(self, seq_length = 10) :
        
        # General Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
        # Data Handling 
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz-.#'
        self.char_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        
        self.txt = self.preprocess(pd.read_parquet(sources['drug']))
        
        # Encoding
        self.seq_length = 10
        self.x = []
        self.y = []
    
    def __getitem__(self, index):
        seq_in = self.txt[index:index + self.seq_length]
        seq_out = self.txt[index + self.seq_length]
        x = F.one_hot(torch.tensor([self.char_to_int[char] for char in seq_in]))
        # y = self.char_to_int[seq_out]
        y = torch.tensor(np_utils.to_categorical(self.char_to_int[seq_out] + 1))
        return x.to(self.device), y.to(self.device)
        
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def preprocess(self, df):
        df = df.assign(__brandname=df['__brandname'].str.split(' ')).explode('__brandname')
        df.drop(df[df['__brandname'].str.isspace() == True].index, inplace = True)
        df.drop_duplicates(inplace = True)
        df = pd.DataFrame(df['__brandname'].apply(lambda s:self.transforms(s)))
        return df['__brandname'].str.cat(sep='#')
        
    def transforms(self, txt):
        txt = txt.strip().lower()
        txt = remove_stopwords(txt)
        # transformation 
        pattern = [
            '[^\x00-\x7f]',   # insure file is acii
            '[0-9]',          # remove digits from text
            '[^\w\s]',        # remove punctuation 
            '(-)'         # remove underscores
            ]
        txt = re.sub('|'.join(pattern), '', txt)
        return txt
    
class DrugnameDataset(Dataset):
    # https://www.kaggle.com/code/jonsteve/generating-prescription-drug-brand-names/notebook
    # Web Source Url: https://www.rxassist.org/pap-info/brand-drug-list-print
    def __init__(self, seq_length = 8):
        self.txt = self.preprocess(pd.read_parquet(sources["drug"]))
        self.dataframe = pd.read_parquet(sources['drug'])
        self.seq_length  = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = sorted(set(self.txt))
        self.vocab_size = len(self.vocab)  
    
        self.char_to_idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(self.vocab)}
        
        self.index_to_char = {index: char for index, char in enumerate(self.vocab)}
        self.char_to_index = {char: index for index, char in enumerate(self.vocab)}
        
        self.char_indexes = [self.char_to_index[c] for c in self.txt]
    
    def __getitem__(self, idx):
        return(
            torch.tensor(self.char_indexes[idx:idx + self.seq_length]).to(self.device),
            torch.tensor(self.char_indexes[idx + 1: idx + self.seq_length + 1]).to(self.device)
        )
 
    def preprocess(self, df):
        df = df.assign(__brandname=df['__brandname'].str.split(' ')).explode('__brandname')
        df.drop(df[df['__brandname'].str.isspace() == True].index, inplace = True)
        df.drop_duplicates(inplace = True)
        df = pd.DataFrame(df['__brandname'].apply(lambda s:self.transforms(s)))
        return df['__brandname'].str.cat(sep='#')
        
    def transforms(self, txt):
        txt = txt.strip().lower()
        # txt = remove_stopwords(txt)
        # transformation 
        pattern = [
            '[^\x00-\x7f]', # insure file is acii
            '[0-9]',        # remove digits from text
            '[^\w\s]',      # remove punctuation 
            '(-)'           # remove underscores
            ]
        txt = re.sub('|'.join(pattern), '', txt)
        return txt

    def __len__(self):
        return len(self.char_indexes) - self.seq_length
    
    def params(self):
        return(
            self.vocab_size,
            self.char_to_idx,
            self.idx_to_char 
        )
    
class MarvelCharacters(Dataset):
    def __init__(self, seq_length = 6):
        self.txt = self.preprocess(pd.DataFrame(pd.read_parquet(sources['characters'])['name']))
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # create mapping of unique chars to integers
        self.vocab = sorted(list(set(self.txt)))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.vocab))
        self.int_to_char = dict((i, c) for c, i in enumerate(self.vocab))
        
    def preprocess(self, df):
        df = df.assign(name=df['name'].str.split(' ')).explode('name')
        df = pd.DataFrame(df['name'].apply(lambda s:self.__transform(s)))
        df.drop_duplicates(inplace=True)
        return df['name'].str.cat() 

    def __transform(self, txt):
        pattern = [
            '[^\x00-\x7f]', # insure acii
            '[^\w\s]',      # remove punctuation 
            '(-)'           # remove underscores
            ]
        txt = re.sub('|'.join(pattern), '', txt)
        return txt.lower()
        
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def __getitem__(self, idx):
        dataX = [] #these are our features 
        dataY = [] #this is our target i.e predicted values
        
        # seq_in is our data features
        # the length is defined via the seq_length vaiable
        seq_in = self.txt[idx:idx + self.seq_length]
        # seq_out is our predicted value i.e the value we are trying to predict
        seq_out = self.txt[idx + self.seq_length]
        
        dataX.append([self.char_to_int[char] for char in seq_in])
        dataY.append(self.char_to_int[seq_out])
        X = torch.tensor(dataX, dtype=torch.float32).to(self.device)
        Y = torch.tensor(dataY, dtype=torch.float32).to(self.device)
        return X, Y
    
class Gameofthrone(Dataset):
    def __init__(self, seq_length=8) -> None:
        self.txt = self.load_and_transform()
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create mapping of unique chars to integers
        self.vocab = self.txt
        self.char_to_idx = {char: i for i, char in enumerate(self.txt)}
        self.idx_to_char = {i: char for i, char in enumerate(self.txt)}
        
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        x = self.char_to_idx[self.txt[idx]]
        x = torch.tensor([x])
        x = F.one_hot(x, num_classes = len(self.vocab))
        x = x.type(torch.FloatTensor)
        t = self.char_to_idx[self.txt[idx + (idx < (self.__len__() - 1 ))]]
        t = torch.tensor([t])
        return(x.to(self.device), t.to(self.device))

    def load_and_transform(self):
        try:
            txt = open(sources['got']).read().replace('\n', '').lower()
        except FileNotFoundError as e:
            raiseExceptions(f'method call load_and_transform GOT file not found msg: {e} path:{dir}')
        
        txt = remove_stopwords(txt)
        # transformation 
        pattern = [
            '[^\x00-\x7f]',  # insure file is acii
            '[0-9]',         # remove digits from text
            '[^\w\s]',       # remove punctuation 
            '\s+'
            ]
        txt = re.sub('|'.join(pattern),'',txt).strip()
        return txt
    
    def get_params(self):
        return (self.vocab_size, self.char_to_idx, self.idx_to_char)
    
class MedicalTerms(Dataset):
    def __init__(self, seq_length = 8) -> None:
        self.txt = self.load()
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # create mapping of unique chars to integers
        self.vocab = sorted(list(set(self.txt)))
        self.itos  = dict(enumerate(self.vocab))
        self.stoi  = {k:v for v, k in self.itos.items()}
        
        
        self.word_idx = [self.stoi[w] for w in self.txt]
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.word_idx[idx:idx + self.seq_length]).to(self.device),
            torch.tensor(self.word_idx[idx + 1: idx + self.seq_length + 1]).to(self.device)
        )
    
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def load(self):
        df = pd.DataFrame(pd.read_fwf(sources['medical'], header=None, names=["term"]) \
            ['term'].apply(lambda txt:self.transforms(txt) if isinstance(txt, str) and len(txt) > 4 else ' '))
        df = df[df.term.str.len() > 3]
        return df['term']#.str.cat(sep='#')

    def transforms(self, txt):
        pattern = [
            # '[^\x00-\x7f]',     # insure acii
            '[^\w\s]',            # remove punctuation 
            '(-)',                # remove underscores
            '[0-9]'               # remove digits from text
            # '[a-zA-Z]{3,7}'     # inteneded to remove abbrevations
            ]
        txt = re.sub('|'.join(pattern), '', txt)
        return txt
    
if __name__=='__main__':
    nadaq = Nasdaq()
    dataloader = DataLoader(dataset = nadaq, batch_size=10) 
    for batch, (x, y) in enumerate(dataloader):
        print(f'x:{x} y:{y}')