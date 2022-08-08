import argparse
from curses.ascii import isspace
import re
import os
import numpy as np

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Adam
from  torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import DrugnameDataset , Gameofthrone
from models.charlstm import CharLSTM, CharLSTM3 , CharLSTM2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default = 5) # epoch should default to 20
    parser.add_argument('--batch_size', type=int, default = 100)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--seq_length', type=int, default = 10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--device", type=str, default=(torch.device('cuda' if torch.cuda.is_available() else "cpu")))
    parser.add_argument('--logdir', type=str, default='logs/')
    parser.add_argument('--model_dir',  type=str, default='modeldir/')
    parser.add_argument('--datasource', type=str, default='drug')
    parser.add_argument('--save_model',  type=bool, default=False)
    parser.add_argument('--sample_size', type=int, default=1)
    parser.add_argument('--amount_of_names_to_generate', type=int, default=10)
    args = parser.parse_args()
    
    drug = DrugnameDataset(args.seq_length)
    model = CharLSTM3(drug)
    
    train(drug, model, args)
    
    generated_names = []
    while len(generated_names) < args.amount_of_names_to_generate:
        name = predict(dataset = drug, model=model, args = args, text=get_randon_names(drug.dataframe,1))
        
        if realistic(name):
            print(f'{name}')
            generated_names.append(name)
    
    # print(f'{predict(dataset = drug, model = model, args=args, text="Abilify".lower())}')
    
def train(dataset, model, args):
    model.train()
    
    dataloader = DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle=False) 
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    writer = SummaryWriter(args.logdir)
    
    for epoch in range(args.epochs):
        state_h, state_c = model.init_state(args.seq_length)
        
        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2),y)
            writer.add_scalar("loss | train", loss, epoch)
            
            state_h = state_h.detach()
            state_c = state_c.detach()
            
            loss.backward()
            optimizer.step()
        
            print(f'epoch: {epoch + 1} | batch: {batch} | loss: {loss.item()}')
    writer.flush()
    save(args, model, loss)
               
def predict(dataset, model, text, args, next_char = 8):
    model.eval()
    chars = ''
    txt = [l  for c in text.lower().split() for l in c]
    state_h, state_c = model.init_state(len(text))
    
    for i in range(0, next_char):
        x = torch.tensor([dataset.char_to_index[c] for c in txt]).to(args.device)
        x = x.unsqueeze_(0)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_char_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_char_logits.to('cpu'), dim=0).detach().numpy()
        char_indx = np.random.choice(len(last_char_logits), p = p)
        char = dataset.index_to_char[char_indx]
        chars += char

    return chars

def get_randon_names(df: pd.DataFrame, sample_size:int):
    txt = df[df.columns[0]].sample(1).str.cat()
    return txt.strip().lower()
    # if txt.isspace() or None:
    #     return df[df.columns[0]].sample(1).str.cat()
    
def realistic(word:str):
    # this regexp matches double letters
    regexp = re.compile(r'(.)\1')
    
    # define what consonants and vowels are
    vowls = 'aeiou'
    cons  = 'bcdfghjklmnpqrstvwxz'

    if re.search(regexp, word):
        return False
    else:
        try: 
            ratio = len([char for char in word if char in cons]) / len([char for char in word if char in vowls]) + 0.001
        except ZeroDivisionError:
            ratio = 0
            
        if ratio >= 2:
                return False
        if ratio <= 0.5:
                return False
        else:
            return True

def save(args, model, loss):
    if args.save_model:    
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
            
        torch.save(model.state_dict(), os.path.join(args.model_dir, f'{args.datasource}-loss-{loss.item()}.pth'))

def load_chkpt(*args, model):
    pass

if __name__=='__main__':
   main()