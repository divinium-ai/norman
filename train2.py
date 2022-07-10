import argparse
from logging import raiseExceptions
import re 

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from models import  char_lstm_module as lstm 

from datasets import dataset as datasources

def main():
    parser = argparse.ArgumentParser()
   
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='drug', help='available datasets: drug | characters | got | medical')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--sequence_length', type=int, default=6, help='length of words to generate')

    # Model parameters
    parser.add_argument('--layers', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005, help='desired learning rate')

    # Training loop params
    parser.add_argument('--epochs', type=int, default=5, help='maxium number of epachs to train for')
    parser.add_argument('--weight_path', type=str, default='/weights/' )
    parser.add_argument('--device', type=str, default=(torch.device('cuda' if torch.cuda.is_available() else "cpu") ))
    
    # Tensorboard
    parser.add_argument('--logdir', type=str, help='run directory for tensorboards log files')
    args = parser.parse_args()
    train(args)
    
def train(args):
    # user the below link for reference in building a standard training loop in torch:
    # https://learning.oreilly.com/library/view/pytorch-pocket-reference/9781492089995/ch05.html#idm45461529327032
    
    dataset = Dataset()
    # Determine which dataset we are using for training run 
    match args.dataset.lower():
        case 'drug':
            dataset = datasources.DrugnameDataset(args.sequence_length)
        case 'characters':
            dataset = datasources.MarvelCharacters(args.sequence_length)
        case 'got':
            dataset = datasources.Gameofthrone(args.sequence_length)
        case 'medical':
            dataset = datasources.MedicalFromFile(args.sequence_length)
        case _:
            raiseExceptions(f'unknown datasouce {arg.source}')
            
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=args.shuffle)  
    criterion  = nn.NLLLoss()
    # TODO model parameters aren't alinged with proper normancluture correct 
    model      = lstm.CharLSTM(dataset_vocab_length=len(dataset.vocab), lstm_size=args.layers, embeddings=args.layers ).to(args.device)
    optimizer  = optim.Adam(model.parameters(), lr = args.lr)
    hidden = lstm.CharLSTM.initHidden()
    
    loss = 0
    # Training Loop
    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            input, targets = batch
            
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            
        # Validation

        # TODO integrate into training loop for feedback
        print(f'Epoch: {epoch} - Training Loss: {loss} - Validation Loss: #TODO:')
    
    # Testing

def check_point():
    path = ''
    filename = ''
    metadata = ''
    
# def train(dataset, model, args):
#     model.train()
    
#     dataloader = DataLoader(dataset, batch_size = args.batch_size)
#     criterion  = nn.CrossEntropyLoss()
#     optimizer  = optim.Adam(model.parameters(), lr = 0.001)
    
#     for epoch in range(args.max_epoch):
#         state_h, state_c = model.init_state(args.sequence_length)
        
#         for batch, (x, y) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             y_pred, (state_h, state_c) = model(x, (state_h, state_c))
#             loss = criterion(y_pred.transpose(1, 2), y)
            
#             state_h = state_h.detach()
#             state_c = state_c.detach()
            
#             loss.backward()
#             optimizer.step()
#             print(f'epoch: {epoch} | batch: {batch} | loss: {loss.item()}')
            
def realistic(word):
    # this regexp matches double letters
    regexp = re.compile(r'(.)\1')
    
    # define what consonants and vowels are
    vowls = 'aeiou'
    cons  = 'bcdfghjklmnpqrstvwxz'

    if re.search(regexp, word):
        return False
    
    ratio = len([char for char in word if char in cons]) / len([char for char in word if char in vowls]) + 0.001
    if ratio >= 2:
            return False
    if ratio <= 0.5:
            return False
    else:
            return True
        
# def predict(dateset, model, text, next_word = 6):
#     model.eval()
    
#     state_h, state_c = model.init_state()

if __name__=='__main__':
    main()