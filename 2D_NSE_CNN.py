'''This is an autoencoder for training on data from a 2D Navier-Stokes integrator. We test different latent dimenions, which in a CNN
means the number of features in the latent space. The features themselves are most likely multiple dimensional, so this code can be 
improved by saving the features in the latent space and performing an inexpensive PCA analysis on each feature to estimate the actual
dimensionaliy of the system. This code ensures decreasing evaluation losses by throwing out weights that lead to undesirable losses and 
saving (if save_weights = True) and reusing (if reload_weights = True) those that produce desirable results.'''

import os
from os import makedirs
from os.path import exists
import torch
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import (Linear, ReLU, MSELoss, Sequential, Conv1d, Conv2d, Conv3d, Sigmoid, MaxUnpool2d, Dropout2d, AvgPool2d,
                      MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten, Unflatten, ConvTranspose2d)
from torch.optim import Adam

# Creating custom data set from folder
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
           

# Tunable Parameters
features = 10 # latent_dim
kernel = 3
stride = 1
padding = 1

save_weights = False # Save weights to computer
save_features = False
reload_weights = False # Reload weights from previous latent_dim, not saved on computer

state_dictt = 0


# Define the CNN
def CNN():
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            
            '''This is a stacked autoencoder (autoencoder1 = encode1 + decode1, autoencoder2 = encode2 + decode2). The middle of the 
            two autoencoders is decode1 and encode2. This operator 'P' maps vectors in the first latent space to the second. The two
            latent spaces should ideally be the same, therefore P is the idenity operator and P^2 - P = 0. This is the condition that 
            we will enforce to obtain better encodings.'''
            
            # Output size of nxn matrix with fxf kernel and p padding is
            # (n + 2p - f + 1)x(n + 2p - f + 1)
            # n' = 512 + 2p - f + 1 = 510
            
            self.encode1 = Sequential(
                Conv2d(1, int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), features, kernel, stride=stride, padding=padding),
                Dropout2d(0.2))
                
                
                # Linear Section
                # Flatten(start_dim=0),
                # Linear(int(10 * 64 * 50 * 50), 50),
                # Sigmoid(),
                # Linear(50, 50))
            
            
            # DECODER
            self.decode1 = Sequential(
                # Linear Section
                # Linear(50, 50),
                # Sigmoid(),
                # Linear(50, int(10 * 64 * 50 * 50)),
                # Unflatten(dim=0, unflattened_size=(10, 64, 50, 50)),
            
               
                Conv2d(features, int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), 1, kernel, stride=stride, padding=padding))
            
            
            self.encode2 = Sequential(
                Conv2d(1, int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), features, kernel, stride=stride, padding=padding),
                Dropout2d(0.2))
            
            
            # DECODER
            self.decode2 = Sequential(
                Conv2d(features, int(features/2), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/2), int(features/4), kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(int(features/4), 1, kernel, stride=stride, padding=padding))

            
        def forward(self, x):
            
            encoded1 = self.encode1(x)
            if save_features:
                # Save features for dimensionality analysis
                for i in range(features):
                  matrix = pd.DataFrame(encoded1[0][i].detach().numpy())
                  matrix.to_csv(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Features/feature{i}.csv')
            decoded1 = self.decode1(encoded1)
            encoded2 = self.encode2(decoded1)
            decoded2 = self.decode2(encoded2)
            return (encoded1, encoded2, decoded2)

    def evaluate(model):
        model.eval()
        running_eval_loss = 0
        
        for input in eval_loader:
            input = torch.unsqueeze(input, 1)
            eval_prediction = model(input)
            loss_outputs = criterion(input, eval_prediction[2])
            loss_identity = criterion(eval_prediction[0], eval_prediction[1])
            running_eval_loss += loss_identity.item()
        
        epoch_eval_loss = running_eval_loss / len(eval_loader)
        eval_loss.append(epoch_eval_loss)
        print('evaluation loss: ', epoch_eval_loss, step)
        
        
    model = Net()
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    if step == 2:
        model.load_state_dict(state_dictt, strict=False)
        model.eval()
    
    # Reuse weights for more consistent graphs
    if latent_dim == 8 and reload_weights:
        weight_dict[latent_dim] = model.state_dict()  # Store weights in weight_dict
        
    elif latent_dim > 8 and reload_weights:
        weight_dict[latent_dim] = model.state_dict()  # Store weights in weight_dict
        pretrained_dict = weight_dict[latent_dim - 8]  # Collect weights from weight_dict
        model_dict = model.state_dict()
        
        for k, v in pretrained_dict.items():
            model_dict[k] = v  # Load recycled weights into current model
        model.state_dict().update(model_dict)
            

    # Training Loop
    for e in range(2):
        model.train()
        running_loss = 0
        
        index = 0
        for input in training_loader:
            input = torch.unsqueeze(input, 1)
            prediction = model(input)
            loss_outputs = criterion(input, prediction[2])
            loss_identity = criterion(prediction[0], prediction[1])
            optimizer.zero_grad()
            # loss_outputs.backward(retain_graph=True)
            loss_identity.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_identity.item()
            index += 1
            
        epoch_loss = running_loss/len(training_loader)
        training_loss.append(epoch_loss)
        scheduler.step(epoch_loss)
        
        evaluate(model)
        
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])
        print(f'epoch: {e+1}')
        # print('training loss: ', epoch_loss)
        print('latent dimension: ', latent_dim)
        print('subdomain: ', subdomain)
    evalDict[latent_dim] = eval_loss[-1]
    
    if save_weights and latent_dim == latent_dims[-1]:  # Save weights upon last latent_dim (or upon other choice)
        torch.save(model.state_dict(), f'/Users/darinmomayezi/Desktop/Vorticity_single_period/weights{subdomain}.pth')
        
    if step == 1:
        state_dictt = model.state_dict()
        
            


# Run CNN and plot results for each subdomain in vorticity data
for subdomain in range(2, 65):
    
    omega_sub = f'omega{subdomain}_sub'  # key to access data in dictionary when using loadmat
    
    # Load Data
    training_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train/omega_names_subdomain.csv',
                transform=normalize)
    eval_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_test',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_test/omega_names_subdomain_test.csv',
                transform=normalize)
    training_loader = DataLoader(training_data, batch_size=10)
    eval_loader = DataLoader(eval_data, batch_size=10)
    
    # Store weights for reuse
    weight_dict = {}

    losses = []
        
    training_loss = []  
    eval_loss = []
    evalDict = {}


    latent_dims = [8, 16, 24, 32, 40, 48, 56, 64]
    for latent_dim in latent_dims:
        features = latent_dim
        
        if latent_dim not in evalDict:
            evalDict[latent_dim] = 0 
            
        for step in range(1, 3):  # Step 1 learns identity, step 2 learns decoding
            
            CNN()
        
        error_increasing = True
        while error_increasing:  # Make sure losses are sufficiently decreasing
            if latent_dim == latent_dims[0]:  # skip first loss
                losses.append(evalDict[latent_dim])
                error_increasing = False
            
            
            elif (evalDict[latent_dim] / evalDict[int(latent_dim - 8)]) < 4:
                losses.append(evalDict[latent_dim])
                error_increasing = False
            
            elif (evalDict[latent_dim] / evalDict[int(latent_dim - 8)]) > 4:
                CNN()
                
        if latent_dim == latent_dims[-1]: # Plot results after running the last latent dimension
            # Graph results
            plt.figure(subdomain)  # creates a new plt canvas
            plt.plot(latent_dims, losses, marker='o')
            plt.xticks(latent_dims)
            
            # Label loss if at least 3e-5 smaller than previous
            yticks = []
            yticks_labels = []
            for index, loss in enumerate(losses):
                if index == 0:
                    yticks.append(losses[index])
                    yticks_labels.append(str(round(losses[index], 8)))
                elif (losses[index - 1] / losses[index]) >= 2:  # Label if at least 5x smaller than previous loss
                    yticks.append(losses[index])
                    yticks_labels.append(str(round(losses[index], 8)))
            plt.yticks(yticks, yticks_labels)
            
            plt.xlabel('Latent Features')
            plt.ylabel('Evaluation loss')
            plt.title(f'Convlutional Autoencoder - 2D NSE - Subdomain {subdomain}')
            plt.show()

            # plt.savefig(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_test/AE_2D_NSE_subdomain{subdomain}.png')
    break
