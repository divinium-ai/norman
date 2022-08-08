import argparse
from logging import raiseExceptions
import re 
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from models import charlstm as CharModel
from torch.utils.tensorboard import SummaryWriter
from datasets import dataset as datasources


def main():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--dataset', type=str, default='drug', help='available datasets: drug | characters | got | medical')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--sequence_length', type=int, default=6, help='length of words to generate')
    parser.add_argument('--workers', type=int, default=(os.cpu_count()//2), help="We'll default to have the avaiable cpu's")
    # Model parameters
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=256)
    # Training loop params
    parser.add_argument('--lr', type=float, default=0.0005, help='desired learning rate')
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
    # https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/961af79f65c54f3077e37ef6e7e60eef/trainingyt.ipynb#scrollTo=oLo3ivjt7u9k
    # https://www.youtube.com/watch?v=jF43_wj_DCQ
    # https://github.com/AmanDaVinci/DeepOrigins/blob/master/notebooks/04_Rebuilding-PyTorch-Essentials.ipynb
    
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
            raiseExceptions(f'unknown datasouce {args.source}')
   
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=args.shuffle)  
    criterion = nn.CrossEntropyLoss()
    model = CharModel.CharLSTM(input_size=dataset.vocab_size, hidden_size=args.hidden_size, output_size=dataset.vocab_size, device=args.device).to(args.device)
    # model = CharModel.CharLSTM(params[0], hidden_size=args.hidden_size, device=args.device, layers=args.layers).to(args.device)
    optimizer  = optim.Adam(model.parameters(), lr = args.lr)
    

    # Training Loop
    for epoch in range(args.epochs):
        model.train(mode=True)
        for idx, (x, y) in enumerate(dataloader):
            score, h, c = model(x, (h, c))
            loss = criterion(score.squeeze(dim = 1), y.squeeze(dim = 1))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if epoch % 500 == 0:
                print('-' * 80)
                print(f'{epoch}: {loss}')
                model.train(mode=False)
                print(f'{model.sample(x[0])}')
        print(f'epochs {epoch}')
            
    
    # Training Loop
    # for epoch in range(args.epochs):
    #     model.train(True)        
    #     for idx, (x, y) in enumerate(dataloader):
            
    #         # Ensuring we are using gpu
    #         x, y = x.to(args.device), y.to(args.device)
            
    #         # forward
    #         scores = model(x, y)
    #         lossfn = criterion(scores, y)
            
    #         #backward step
    #         optimizer.zero_grad()
    #         lossfn.backward()
            
    #         #Gradient Descent step
    #         optimizer.step()
            

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

if __name__=='__main__':
   main()