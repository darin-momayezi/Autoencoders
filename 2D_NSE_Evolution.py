'''This script is meant to test the predictive capability of a CNN model on the 2-D NSE integrator data (or other data). The Autoencoder
learns two steps sequentially: first the encoder learns an identity operator that maps the data to a low dimensional latent space and 
then the decoder learns an envolution operator that maps the data from the latent space to a specified time in the future. 
The Autoencoder is first trained to learn the identity operator, the weights are saved and then '''

import os
import torch
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import (Linear, ReLU, MSELoss, Sequential, Conv2d, Dropout2d, Sigmoid,
                      MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten, Unflatten, ConvTranspose2d)
from torch.optim import Adam

# Custom Dataset
class NSEDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        matrix_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        matrix = np.array(loadmat(matrix_path)[omega_sub])
        if self.transform:
            matrix = self.transform(matrix)
        return torch.tensor(matrix, dtype=torch.float32)

# Normalization transform
def normalize(matrix):
    norm = np.linalg.norm(matrix[0], 1)
    matrix = matrix/norm
    return matrix

# Encoded dimension
features = 10 # latent_dim
kernel = 3
stride = 1
padding = 1


save_weights = False

# Define the CNN
def CNN():
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            
            # Output size of nxn matrix with fxf kernel and p padding is
            # (n + 2p - f + 1)x(n + 2p - f + 1)
            # n' = 512 + 2p - f + 1 = 510
            
            self.encode = Sequential(
                Conv2d(1, int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), features, kernel, stride=stride, padding=padding),
                Dropout2d(0.2))
            
            
            # DECODER
            self.decode = Sequential(
                Conv2d(features, int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), 1, kernel, stride=stride, padding=padding))

            
        def forward(self, x):
            encoded = self.encode(x)
            # for i in range(features):
            #     matrix = pd.DataFrame(encoded[0][i].detach().numpy())
            #     matrix.to_csv(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Features/feature{i}.csv')
            decoded = self.decode(encoded)
            return decoded

    model = Net()
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
    
    def encoder_train(model, criterion, optimizer):  # Encoder learns the identity operator
        model.train()
        running_loss = 0
        
        for input in training_loader:
            input = torch.unsqueeze(input, 1)
            prediction = model(input)
            loss = criterion(input, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss/len(training_loader)
        encoder_training_loss.append(epoch_loss)
        
    def encoder_evaluate(model, criterion):
        model.eval()
        running_loss = 0
        
        for input in eval_loader:
            input = torch.unsqueeze(input, 1)
            prediction = model(input)
            loss = criterion(input, prediction)
            running_loss += loss.item()
        
        epoch_eval_loss = running_loss / len(eval_loader)
        encoder_eval_loss.append(epoch_eval_loss)
        print('encoder evaluation loss: ', epoch_eval_loss)
        
        
        
        
    def decoder_train(model, criterion, optimizer):  # Decoder learns time evolution operator
        model.train()
        running_loss = 0
        
        index = 0
        previous = 0
        for input in training_loader:
            
            if index == 0:
                previous = input
                prediction = model(torch.unsqueeze(previous, 1))
                
            elif index > 0:
                prediction = model(torch.unsqueeze(previous, 1))  # Evolve previous state to current time step
            
            loss = criterion(torch.unsqueeze(input, 1), prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            previous = input
            index += 1
                
        epoch_loss = running_loss/len(training_loader)
        decoder_training_loss.append(epoch_loss)    
        
    def decoder_evaluate(model, criterion):
        model.eval()
        running_eval_loss = 0
        
        index = 0
        previous = 0
        for input in eval_loader:
            
            if index == 0:
                previous = input
                prediction = model(torch.unsqueeze(previous, 1))
                
            elif index > 0:
                prediction = model(torch.unsqueeze(previous, 1))  # Evolve previous state to current time step
            
            loss = criterion(torch.unsqueeze(input, 1), prediction)
            running_eval_loss += loss.item()
            previous = input
            index += 1
            
        epoch_eval_loss = running_eval_loss / len(eval_loader)
        decoder_eval_loss.append(epoch_eval_loss)
        print('decoder evaluation loss: ', epoch_eval_loss)
            
    for step in range(1, 3):  # Step 1 learns identity, step 2 learns time evolution
        
        if step == 1:
            for epoch in range(200):
                encoder_train(model=model, criterion=criterion, optimizer=optimizer)
                encoder_evaluate(model=model, criterion=criterion)
            
        if step == 2: 
            # Remove gradient from encoding weights/hold constant
            for i, param in enumerate(model.named_parameters()):
                name = param[0]
                if 'encode' in name:
                    param[1].requires_grad = False
                    
            for epoch in range(200):
                decoder_train(model=model, criterion=criterion, optimizer=optimizer)
                decoder_evaluate(model=model, criterion=criterion)
                
            for i, param in enumerate(model.named_parameters()):
                name = param[0]
                if 'encode' in name:
                    param[1].requires_grad = True



for subdomain in range(1, 2):
    
    omega_sub = f'omega{subdomain}_sub'  # key to access data in dictionary when using loadmat
    
    # Load Data
    training_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train/omega_names_subdomain.csv',
                transform=normalize)
    eval_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_test',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_test/omega_names_subdomain_test.csv',
                transform=normalize)
    training_loader = DataLoader(training_data, batch_size=1)
    eval_loader = DataLoader(eval_data, batch_size=1)
    
    
    encoder_training_loss = []
    decoder_training_loss = []
    encoder_eval_loss = []
    decoder_eval_loss = []
    evalDict = {}
    latents = [32]
    
    for latent_dim in latents:
        features = latent_dim
        if latent_dim not in evalDict:
            evalDict[latent_dim] = 0 
            
        CNN()  
        
    avg_encoder_loss = sum(encoder_eval_loss) / len(encoder_eval_loss)
    avg_decoder_loss = sum(decoder_eval_loss) / len(decoder_eval_loss)
    
    
    plt.plot(subdomain, avg_encoder_loss, marker='o', label='encoder/identity')
    plt.bar(subdomain, avg_decoder_loss, marker='^', label='decoder/evolution')
    plt.xlabel('Subdomain')
    plt.ylabel('Average Evaluation Loss')
    plt.title('Encoder/Decoder Loss for Subdomain')
    
plt.legend(loc='upper right') 
plt.show()
    
