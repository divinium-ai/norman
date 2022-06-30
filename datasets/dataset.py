import re
from logging import raiseExceptions


import numpy as np
import pandas as pd 
from gensim.parsing.preprocessing import remove_stopwords

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from keras.utils import np_utils

sources = {
    'Drug':'datasets/data/Drugnames/drugnames-20220512-parquet.gzip',
    'Drug_txt': 'datasets/data/Drugnames/drugname.txt',
    'Characters': 'datasets/data/marvels/characters-20220517-parquet.gzip',
    'GOT': 'datasets/data/got/gameofthrones.txt',
    'MEDICAL': 'datasets/data/medical/wordlist.txt'
}
class DrugnameDataset(Dataset):
    # https://www.kaggle.com/code/jonsteve/generating-prescription-drug-brand-names/notebook
    # Web Source Url: https://www.rxassist.org/pap-info/brand-drug-list-print
    def __init__(self, seq_length = 8):
        self.txt = self.preprocess(pd.read_parquet(sources["Drug"]))
        self.seq_length  = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create mapping of unique chars to integers
        self.vocab = sorted(list(set(self.txt)))
        self.itos  = dict(enumerate(self.vocab))
        self.stoi  = {k:v for v, k in self.itos.items()}
        
        self.word_idx = [self.stoi[w] for w in self.txt]
        
    def __getitem__(self, idx):        
        dataX = []
        dataY = []
        seq_in  = self.txt[idx:idx + self.seq_length]
        seq_out = self.txt[idx + self.seq_length]
        dataX.append([self.stoi[char] for char in seq_in])
        dataY.append(self.stoi[seq_out])
        
        # X is the input data (time series of 10-character strings)
        X = torch.tensor(np.reshape(dataX, (len(dataX), self.seq_length, 1)))
        # y is the output data (the 11th character to be predicted from the preceding 10)
        y = torch.tensor(np_utils.to_categorical(dataY))
        return  X.to(self.device), y.to(self.device)
    
    def preprocess(self, df):
        df = df.assign(__brandname=df['__brandname'].str.split(' ')).explode('__brandname')
        df.drop(df[df['__brandname'].str.isspace() == True].index, inplace = True)
        df.drop_duplicates(inplace = True)
        df = pd.DataFrame(df['__brandname'].apply(lambda s:self.transforms(s)))
        return df['__brandname'].str.cat(sep= '#')
        
    def transforms(self, txt):
        txt = txt.strip().lower()
        txt = remove_stopwords(txt)
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
        return len(self.txt) - self.seq_length
    
class MarvelCharacters(Dataset):
    def __init__(self, seq_length = 6):
        self.txt = self.preprocess(pd.DataFrame(pd.read_parquet(sources['Characters'])['name']))
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
        return len(self.char_txt) - self.seq_length
    
    def __getitem__(self, idx):
        dataX = [] #these are our features 
        dataY = [] #this is our target i.e predicted values
        
        # seq_in is our data features
        # the length is defined via the seq_length vaiable
        seq_in = self.char_txt[idx:idx + self.seq_length]
        # seq_out is our predicted value i.e the value we are trying to predict
        seq_out = self.char_txt[idx + self.seq_length]
        
        dataX.append([self.char_to_int[char] for char in seq_in])
        dataY.append(self.char_to_int[seq_out])
        X = torch.tensor(dataX, dtype=torch.float32).to(self.device)
        Y = torch.tensor(dataY).to(self.device)
        return X, Y
    
class Gameofthrone(Dataset):
    def __init__(self, seq_length=8) -> None:
        self.txt = self.load_and_transform()
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create mapping of unique chars to integers
        self.vocab_size = len(self.txt)
        self.char_to_idx = {char: i for i, char in enumerate(self.txt)}
        self.idx_to_char = {i: char for i, char in enumerate(self.txt)}
    
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.char_to_idx[self.txt[idx]]
        x = torch.tensor([x])
        x = F.one_hot(x, num_classes = self.vocab_size)
        x = x.type(torch.FloatTensor)
        t = self.char_to_idx[self.txt[idx + (idx < (self.__len__() - 1 ))]]
        t = torch.tensor([t])
        return(x.to(self.device), t.to(self.device))

    def load_and_transform(self):
        try:
            txt = open(sources['GOT']).read().replace('\n', '').lower()
        except FileNotFoundError as e:
            raiseExceptions(f'method call load_and_transform GOT file not found msg: {e} path:{dir}')
        
        txt = remove_stopwords(txt)
        # transformation 
        pattern = [
            '[^\x00-\x7f]', # insure file is acii
            '[0-9]',        # remove digits from text
            '[^\w\s]',       # remove punctuation 
            '\s+'
            ]
        txt = re.sub('|'.join(pattern),'',txt).strip()
        return txt
    
    def get_params(self):
        return (self.vocab_size, self.char_to_idx, self.idx_to_char)
    
class MedicalFromFile(Dataset):
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
            torch.tensor(self.word_idx[idx + 1]).to(self.device)
        )
    
    def __len__(self):
        return len(self.txt) - self.seq_length
    
    def load(self):
        df = pd.DataFrame(pd.read_fwf(sources['MEDICAL'], header=None, names=["term"]) \
            ['term'].apply(lambda txt:self.transforms(txt) if isinstance(txt, str) and len(txt) > 3 else ' '))
        df = df[df.term.str.len() > 3]
        return df['term'].str.cat(sep='#')

    def transforms(self, txt):
        pattern = [
            '[^\x00-\x7f]',     # insure acii
            '[^\w\s]',          # remove punctuation 
            '(-)',              # remove underscores
            '[0-9]'             # remove digits from text
            ]
        txt = re.sub('|'.join(pattern), '', txt)
        return txt.lower()
    
    
