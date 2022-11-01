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
from torch.optim import Adam, SGD

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

# # High dimensional input
# idx = 0
# previous = 0
# new_training_data = []
# count = 0
# for input in training_loader:
#     # print(input[0])  # 50x50
#     count += 1
#     if idx == 0:  # skip first matrix
#         idx += 1
#         previous = input[0]
        
#     elif (idx + 1) % 2 == 0:  # concatenate current matrix to previous
#         row_idx = 0
#         new_input = []
#         for row in input[0]:
#             new_row = torch.tensor(np.concatenate((row.numpy(), previous[row_idx].numpy())))
#             new_input.append(new_row)
#             row_idx += 1
#         new_training_data.append(new_input)
#         idx += 1
        
#     elif (idx + 1) % 2 == 1:
#         previous = input[0]
#         idx += 1

# training_data = Dataset(new_training_data)
# training_loader = DataLoader(training_data)
# print(next(iter(training_loader)).shape)
        

            

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
                ReLU(),
                Conv2d(8, 16, kernel, stride=stride, padding=padding),
                AvgPool2d(2),
                
                
                # Linear Section
                Flatten(start_dim=0),
                Linear(int(16 * 25 * 25), 50),
                ReLU(),
                Linear(50, 25))
            
            
            # DECODER
            self.decode = Sequential(
                # Linear Section
                Linear(25, 50),
                ReLU(),
                Linear(50, int(16 * 50 * 50)),
                Unflatten(dim=0, unflattened_size=(16, 50, 50)),
            
               
                Conv2d(16, 8, kernel, stride=stride, padding=padding),
                ReLU(),
                Conv2d(8, 1, kernel, stride=stride, padding=padding))

            
        def forward(self, x):
            
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded

    def evaluate(model):
        model.eval()
        running_eval_loss = 0
        
        for input in eval_loader:
            # input = torch.unsqueeze(input, 1)
            eval_prediction = model(input)
            losse = criterion(input, eval_prediction)
            running_eval_loss += losse.item()
        
        epoch_eval_loss = running_eval_loss / len(eval_loader)
        eval_loss.append(epoch_eval_loss)
        print('evaluation loss: ', epoch_eval_loss)
        
        
    model = Net()
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    # print(model)
            

    # Training Loop
    for e in range(10):
        model.train()
        running_loss = 0
        
        index = 0
        for input in training_loader:
            input = torch.unsqueeze(input[0], 0)
            prediction = model(input)
            loss = criterion(input, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if e >= 0 and index == 0:
                print(input)
                print(prediction)
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
        torch.save(model.state_dict(), '/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/2D_NSE/Vorticity_test/2D_NSE_CNN_weights.pth')


    no_of_layers=0
    conv_layers=[]
    
    model_children=list(model.children())
    
    # for child in model_children:
    #     if type(child)==Conv2d:
    #         no_of_layers+=1
    #         conv_layers.append(child)
    #         print(child)
    #     elif type(child)==Sequential:
    #         for layer in child.children():
    #             if type(layer)==Conv2d:
    #                 no_of_layers+=1
    #                 conv_layers.append(layer)
    # print(conv_layers)

# Run CNN and plot results
training_loss = []  
eval_loss = []
latent_dims = [30]
evalDict = {}
for latent_dim in latent_dims:
    features = latent_dim
    if latent_dim not in evalDict:
        evalDict[latent_dim] = 0 
        
    CNN()  
    
latent_dims = []
losses = []
    
for key, value in evalDict.items():
    
    latent_dims.append(key)
    losses.append(value)
   
plt.plot(latent_dims, losses, marker='o')
# plt.show()
