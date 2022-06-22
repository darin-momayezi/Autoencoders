'''A generic autoencoder architecture to start.'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


##### Parameters #####

input_size = 100
hidden1 = int(input_size/10)
latent_dim = 7
epochs = 100
lr = 0.001  # learning rate
batch_size = 20


##### Data #####

df = pd.read_csv('/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/1D_PDEs/1Dcoarsegrid.csv')
dfT = df.T

train_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[:int(0.7*df.shape[0])])))
eval_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[int(0.7*df.shape[0]):])))
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False)


##### NN ######

def autoencoder():
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_size, hidden1, bias=True, dtype=torch.float32),
                                         nn.ReLU(),
                                         nn.Linear(hidden1, latent_dim, bias=True, dtype=torch.float32))
            
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden1, bias=True, dtype=torch.float32),
                                         nn.ReLU(),
                                         nn.Linear(hidden1, input_size, bias=True, dtype=torch.float32))
            
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
        
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    
    
    train_loss = []
    eval_loss = []
    
    def evaluate(model, eval_loader):
        model.eval()
        running_eval_loss = 0
        
        for i, x in enumerate(eval_loader):
            input_eval = torch.Tensor(x[0][0])
            recone = model(input_eval)
            losse = criterion(input_eval, recone)
            running_eval_loss += losse.item()
        eval_loss.append(running_eval_loss / len(eval_loader))
        print(f"Evaluation Loss: {round(running_eval_loss / len(eval_loader), 20), latent_dim}")

        
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0
        print(f'Epoch: {epoch +1}')
        
        for i, y in enumerate(train_loader):
            input_train = torch.Tensor(y[0][0])
            recont = model(input_train)
            losst = criterion(input_train, recont)
            optimizer.zero_grad()
            losst.backward()
            optimizer.step()
            running_train_loss += losst.item()
        
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])
        train_loss.append(running_train_loss / len(train_loader))
        evaluate(model, eval_loader)
        print(f'Training Loss: {round(running_train_loss / len(train_loader), 20)}')
    evalDict[m].append(eval_loss[-1])
        
   
evalDict = {}    
for i in range(1): 
    for m in range(1, latent_dim+1):  # Dimension of manifold

        latent_dim = m
        
        if latent_dim not in evalDict:
            evalDict[latent_dim] = []
            
        autoencoder()
    

##### Plot results #####
ranging = []
errors = []
for key, values in evalDict.items():
    for value in values:
        ranging.append(key)
        errors.append(value)
plt.plot(ranging, errors, marker='o')
plt.xticks(ranging)
plt.xlabel('Latent Space Dimension')
plt.ylabel('Evaluation Loss')
plt.title('tmax=50000, t=10000, xmax=1000, x=100, k=0.01, q=100')
plt.show()
