from logging import raiseExceptions
import re

from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
import pandas as pd 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


sources = {
    'Drug':'datasets/data/Drugnames/drugnames-20220512-parquet.gzip',
    'Drug_txt': 'datasets/data/Drugnames/drugname.txt',
    'Characters': 'datasets/data/marvels/characters-20220517-parquet.gzip',
    'GOT': 'datasets/data/got/gameofthrones.txt'
}
    
class DrugnameDataset(Dataset):
    
    """
    
    Port from Keras to torch 
    https://www.kaggle.com/code/jonsteve/generating-prescription-drug-brand-names/notebook
    
    """

    def __init__(self, dir):
        try:
            self.txt = open(dir).read().lower()
        except FileNotFoundError as e:
            raise Exception(f'__init__ no file found msg: {e}')
        self.seq_length = 10
        self.dataX = []
        self.dataY = []
        
    def __len__(self):
        return len(self.txt)
    
    def __getitem__(self, idx):
        return self.txt[idx]
    
class MarvelCharacters(Dataset):
    def __init__(self, dir):
        try:
            self.data_frame = pd.DataFrame(pd.read_parquet(dir)['name'])
        except FileNotFoundError as e:
            raise Exception(f'__init__ no file found msg: {e}')
        self.characters = self.preprocessing()
        
    def preprocessing(self):
        tmpdf = self.data_frame.apply(lambda x:x.lower() if type(x) == str else 'x')
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
class Gameofthrone(Dataset):
    def __init__(self,device) -> None:
        self.chars, self.txt = self.load_and_transform()
        self.vocab_size = len(self.chars)
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        self.device = device
    
    def __len__(self):
        return len(self.txt)
    
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
            '[^\w\s]'       # remove punctuation 
            ]
        txt = re.sub('|'.join(pattern),'',txt).strip()
        return sorted(set(txt)), txt
    
    def get_params(self):
        return (self.vocab_size, self.char_to_idx, self.idx_to_char)
    
# if __name__ == '__main__':
#     # GOT Dataset
#     torch.multiprocessing.set_start_method('spawn')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     GOT_DATASET = Gameofthrone(device=device)
#     dataloader = DataLoader(GOT_DATASET, batch_size=18, shuffle=True, num_workers=2, pin_memory=False)
#     for i, batch in enumerate(dataloader):
#         print(i, batch)
                
    # dataset = DrugnameDataset(dir=sources['Drug_txt'])
    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=2, pin_memory=True)
    # for i, batch in enumerate(dataloader):
    #     print(i,batch)
        
    # print(len(dataset))
    # print(dataset[:2])
    # print(dataset[5:12])