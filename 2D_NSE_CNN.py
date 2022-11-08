import os
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
           

# Variables
features = 10 # latent_dim
kernel = 3
stride = 1
padding = 1

save_weights = True
save_features = False

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
                
                
                # Linear Section
                # Flatten(start_dim=0),
                # Linear(int(10 * 64 * 50 * 50), 50),
                # Sigmoid(),
                # Linear(50, 50))
            
            
            # DECODER
            self.decode = Sequential(
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

            
        def forward(self, x):
            
            encoded = self.encode(x)
            if save_features:
                # Save features for dimensionality analysis
                for i in range(features):
                  matrix = pd.DataFrame(encoded[0][i].detach().numpy())
                  matrix.to_csv(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Features/feature{i}.csv')
            decoded = self.decode(encoded)
            return decoded

    def evaluate(model):
        model.eval()
        running_eval_loss = 0
        
        for input in eval_loader:
            input = torch.unsqueeze(input, 1)
            eval_prediction = model(input)
            losse = criterion(input, eval_prediction)
            running_eval_loss += losse.item()
        
        epoch_eval_loss = running_eval_loss / len(eval_loader)
        eval_loss.append(epoch_eval_loss)
        print('evaluation loss: ', epoch_eval_loss)
        
        
    model = Net()
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    if latent_dim > 8:
        # Reuse weights for more consistent graphs
        pretrained_dict = torch.load('/Users/darinmomayezi/Desktop/pretrained_weights2.pth')
        model_dict = model.state_dict()
        for k, v in pretrained_dict.items():
            model_dict[k] = v
        model_dict.update(model_dict)
        print(len(pretrained_dict))
        print(len(model.state_dict()))
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
            

    # Training Loop
    for e in range(200):
        model.train()
        running_loss = 0
        
        index = 0
        for input in training_loader:
            input = torch.unsqueeze(input, 1)
            prediction = model(input)
            loss = criterion(input, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if e >= 0 and index == 0:
            #     print(input[0][0])
            #     print(prediction[0][0])
            index += 1
            
        epoch_loss = running_loss/len(training_loader)
        training_loss.append(epoch_loss)
        
        evaluate(model)
        
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])
        print(f'epoch: {e+1}')
        print('training loss: ', training_loss)
    evalDict[latent_dim] = eval_loss[-1]
    if save_weights:
        torch.save(model.state_dict(), '/Users/darinmomayezi/Desktop/pretrained_weights2.pth')


# Run CNN and plot results for each subdomain in vorticity data
for subdomain in range(1, 65):
    
    omega_sub = f'omega{subdomain}_sub'  # key to access data in dictionary when using loadmat
    
    # Load Data
    training_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_data/Vorticity_subdomain{subdomain}_train',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_data/Vorticity_subdomain{subdomain}_train/omega_names_subdomain.csv',
                transform=normalize)
    eval_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_data/Vorticity_subdomain{subdomain}_test',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_data/Vorticity_subdomain{subdomain}_test/omega_names_subdomain_test.csv',
                transform=normalize)
    training_loader = DataLoader(training_data, batch_size=10)
    eval_loader = DataLoader(eval_data, batch_size=10)
    
    training_loss = []  
    eval_loss = []
    evalDict = {}
    
    latent_dims = [8, 16, 24, 32, 48, 64]
    for latent_dim in latent_dims:
        features = latent_dim
        
        if latent_dim not in evalDict:
            evalDict[latent_dim] = 0 
            
        CNN()  
        
    losses = []
    for key, value in evalDict.items():
        losses.append(value)
        
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
            yticks_labels.append(str(round(losses[index], 5)))
        elif (losses[index-1] - losses[index]) > 3e-5:
            yticks.append(losses[index])
            yticks_labels.append(str(round(losses[index], 5)))
    plt.yticks(yticks, yticks_labels)
    
    plt.xlabel('Latent Features')
    plt.ylabel('Evaluation loss')
    plt.title('Convlutional Autoencoder - 2D NSE')
    plt.savefig(f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_data/Vorticity_subdomain{subdomain}_test/AE_2D_NSE_subdomain{subdomain}.png')
    

