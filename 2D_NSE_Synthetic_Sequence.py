import os
import torch
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import (Linear, ReLU, MSELoss, Sequential, Conv2d, Dropout2d,
                      MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Flatten, Unflatten, ConvTranspose2d)
from torch.optim import Adam, sgd

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

# Load training data
training_data = NSEDataset(root_dir='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_train/',
                annotation_file='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_train/omega_names_subdomain.csv',
                transform=normalize)
eval_data = NSEDataset(root_dir='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_test/',
                annotation_file='/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_subdomain_test/omega_names_subdomain_test.csv',
                transform=normalize)
training_loader = DataLoader(training_data)
eval_loader = DataLoader(eval_data)


# Tunable Parameters
features = 10 # latent_dim
kernel = 3
stride = 1
padding = 1


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

            
        def forward(self, x):
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded
        
        
    model = Net()
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = MSELoss()
    model.load_state_dict(
        torch.load(f'/Users/darinmomayezi/Desktop/Vorticity_single_period/weights/evolutionWeights{subdomain}.pth'),
        strict=False)
    model.eval()
    # print(model)
            

    # Training Loop
    running_loss = 0
    
    index = 0
    synthetic = []
    model.train()
    for input in training_loader:
        if index == 0:
            prediction = model(torch.unsqueeze(input, 1))
            synthetic.append(prediction)
        else:
            prediction = model(synthetic[-1])
            synthetic.append(prediction)
        loss = criterion(torch.unsqueeze(input, 1), prediction)
        running_loss += loss.item()
        index += 1
    losses.append(running_loss/len(training_loader))
    print('training loss: ', running_loss/len(training_loader))

losses = []
subdomains = []

latent_dict = {8: [4, 6, 12, 18, 21, 23, 27, 28, 29, 35, 36, 41, 42, 43, 49, 51, 52, 53, 54, 58, 61, 62, 63],
               16: [2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 16, 19, 22, 24, 25, 26, 33, 34, 37, 38, 39, 40, 47, 48, 50, 56, 59, 60, 64],
               24: [1, 9, 17, 30, 31, 32, 44, 45, 57],
               32: [20, 46],
               40: [55]}

for subdomain in range(1, 65):
    
    omega_sub = f'omega{subdomain}_sub'  # key to access data in dictionary when using loadmat
    
    training_data = NSEDataset(root_dir=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train',
                annotation_file=f'/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_single_period/Vorticity_subdomain{subdomain}_train/omega_names_subdomain.csv',
                transform=normalize)
    training_loader = DataLoader(training_data, batch_size=1)
    
    latent_dim = 0
    for latent in latent_dict:
        if subdomain in latent_dict[latent]:
            latent_dim = latent
    
    features = latent_dim
    CNN()   
    subdomains.append(subdomain)
    
plt.plot(subdomains, losses, marker='o')
plt.xlabel("Subdomain")
plt.ylabel('Loss')
plt.title('Synthetic Sequence')
plt.savefig('/Users/darinmomayezi/Desktop/syntheticSequence.png')
plt.show()
