import os

import torch
import torch.nn as nn
from torch.optim import Adam
from  torch.utils.data import DataLoader

from datasets.dataset import Gameofthrone
from models.got import GOTLSTM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH      = 500
HIDDEN_DIM = 250
LEARN_RATE = 0.01
SEQ_LENGTH = 100
LAYERS     = 1

def train_got():
    dataset = Gameofthrone(device=DEVICE)
    dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=False)
    
    vocab_size, char_to_idx, idx_to_char = dataset.get_params()
    model = GOTLSTM(device=DEVICE, char_to_idx=char_to_idx, idx_to_char=idx_to_char, vocab=vocab_size,  
                    hidden_size = 1, hidden_dim = HIDDEN_DIM, layers=LAYERS).to(DEVICE)
    
    lossfn    = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)
    
    h = torch.zeros(LAYERS, 1, HIDDEN_DIM).to(DEVICE)
    c = torch.zeros(LAYERS, 1, HIDDEN_DIM).to(DEVICE)
    i = 0
    for input, target in dataloader:
        scores, h, c = model(input, (h, c))
        loss = lossfn(scores.squeeze(dim=1), target.squeeze(dim=1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % EPOCH == 0:
            print('---' * 90)
            print(i, ': ', loss)
            print(model.sample(input[0]))
            print('---' * 90)
    print(f'batches completed: {i}')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    train_got()