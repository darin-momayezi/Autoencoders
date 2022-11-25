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
        print(matrix_path)
        matrix = np.array(loadmat(matrix_path)['omega1_sub'])
        if self.transform:
            matrix = self.transform(matrix)
        return torch.tensor(matrix, dtype=torch.float32)

# Normalization transform
def normalize(matrix):
    norm = np.linalg.norm(matrix[0], 1)
    matrix = matrix/norm
    return matrix

# Load training data
training_data = NSEDataset(root_dir='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_train/',
                annotation_file='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_train/omega_names_subdomain.csv',
                transform=normalize)
eval_data = NSEDataset(root_dir='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_test/',
                annotation_file='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_test/omega_names_subdomain_test.csv',
                transform=normalize)
training_loader = DataLoader(training_data)
eval_loader = DataLoader(eval_data)
# print(next(iter(training_loader)).shape)  # size of input

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
                Conv2d(1, 8, kernel, stride=stride, padding=padding),
                ReLU(True),
                Conv2d(8, 16, kernel, stride=stride, padding=padding),
                ReLU(True),
                Conv2d(16, 32, kernel, stride=stride, padding=padding),
                ReLU(True),
                Conv2d(32, 64, kernel, stride=stride, padding=padding),
                ReLU(True)
                )
            
            
            # DECODER
            self.decode = Sequential(
            
                # ReLU(True),
                ConvTranspose2d(64, 32, kernel, stride=stride, padding=padding),
                ReLU(True),
                ConvTranspose2d(32, 16, kernel, stride=stride, padding=padding),
                ReLU(True),
                ConvTranspose2d(16, 8, kernel, stride=stride, padding=padding),
                ReLU(True),
                ConvTranspose2d(8, 1, kernel, stride=stride, padding=padding))

            
        def forward(self, x):
            encoded = self.encode(x)
            # for i in range(features):
            #     matrix = pd.DataFrame(encoded[0][i].detach().numpy())
            #     matrix.to_csv(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Features/feature{i}.csv')
            decoded = self.decode(encoded)
            return decoded

    def evaluate(model):
        model.eval()
        running_eval_loss = 0
        
        previous = 0
        index = 0
        for input in eval_loader:
            if index == 0:
                eval_prediction = model(input.unsqueeze(0))
            else:
                eval_prediction = model(previous)
            losse = criterion(input.unsqueeze(0), eval_prediction)
            running_eval_loss += losse.item()
            previous = input.unsqueeze(0)
            index += 1
        
        epoch_eval_loss = running_eval_loss / len(eval_loader)
        eval_loss.append(epoch_eval_loss)
        print('evaluation loss: ', epoch_eval_loss)
        
        
        
    def encoder_train():  # Encoder learns the identity operator
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
        training_loss.append(epoch_loss)
        
    def encoder_evaluate():
        model.eval()
        running_loss = 0
        
        for input in eval_loader:
            input = torch.unsqueeze(input, 1)
            prediction = model(input)
            loss = criterion(input, prediction)
            running_loss += loss.item()
        
        epoch_eval_loss = running_loss / len(eval_loader)
        eval_loss.append(epoch_eval_loss)
        print('evaluation loss: ', epoch_eval_loss)
        
        
        
        
    def decoder_train():  # Decoder learns time evolution operator
        model.train()
        running_loss = 0
        
        index = 0
        previous = 0
        for input in training_loader:
            input = torch.unsqueeze(input, 1)
            
            if index == 0:
                previous = input
                prediction = model(previous)
                loss = criterion(prediction, input)
                
            elif index > 0:
                prediction = model(previous)  # Evolve previous state to current time step
                loss = criterion(input, prediction)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
        epoch_loss = running_loss/len(training_loader)
        training_loss.append(epoch_loss)    
        
    def decoder_evaluate():
        pass   
            
        
        
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
            
    for step in range(2, 3):  # Step 1 learns identity, step 2 learns time evolution
        
        if step == 1:
            for epoch in range(200):
                encoder_train()
            encoder_evaluate()
            
        if step == 2:
            for i, param in enumerate(model.named_parameters()):
                print(param)
            '''Here need to load encoding weights to encoder and turn requires_grad to False so that they don't change.'''
            break
            for epoch in range(200):
                decoder_train()
            decoder_evaluate()











    # Training Loop
    # for e in range(200):
    #     model.train()
    #     running_loss = 0
        
    #     index = 0
    #     previous = 0
    #     for input in training_loader:
    #         if index == 0:
    #             prediction = model(input.unsqueeze(0))
    #             recon = model(input.unsqueeze(0))
    #         else:
    #             prediction = model(previous)
    #             recon = model(input.unsqueeze(0))
    #         evolution_loss = criterion(input, prediction)  # for the decoder
    #         recon_loss = criterion(input, recon)  # for the encoder
    #         optimizer.zero_grad()
    #         evolution_loss.backward()
    #         optimizer.step()
    #         running_loss += recon_loss.item()
    #         previous = input.unsqueeze(0)
    #         index += 1
    #         if e >= 3 and index == 1:
    #             print(input[0][20])
    #             print(prediction[0][0][20])
            
    #     epoch_loss = running_loss/len(training_loader)
    #     training_loss.append(epoch_loss)
        
    #     evaluate(model)
        
    #     for param_group in optimizer.param_groups:
    #         print('lr: ', param_group['lr'])
    #     print(f'epoch: {e+1}')
    #     print('training loss: ', training_loss)
    # evalDict[latent_dim] = eval_loss[-1]
    # if save_weights:
    #     torch.save(model.state_dict(), '/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_test/2D_NSE_CNN_weights.pth')


# Run CNN and plot results
training_loss = []  
eval_loss = []
evalDict = {}
latents = [15]

for latent_dim in latents:
    features = latent_dim
    if latent_dim not in evalDict:
        evalDict[latent_dim] = 0 
        
    CNN()  
    
latent_dims = []
losses = []
    
for key, value in evalDict.items():
    print(key, value)
    latent_dims.append(key)
    losses.append(value)
   
plt.plot(latent_dims, losses, marker='o', label='evaluation loss')
plt.xlabel('Latent Features')
plt.ylabel('Evaluation Loss')
plt.title('Latent Features vs Evaluation Loss')
plt.show()
