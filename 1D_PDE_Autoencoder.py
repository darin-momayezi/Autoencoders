import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from OneDcoarsegrid import T, t, X, x, dx, k, q, F, a, b
import scipy.io as sio


##### Parameters #####

input_size = x  # Number of spatial grid points from OneDcoarsegrid
hidden1 = int(input_size/10)
latent_dim = 4
epochs = 200
lr = 0.001  # Learning rate
batch_size = 30
iterations = 4  # Run autoencoder _ times
shuffle = False  # Don't shuffle temporally sequenced data



##### Data #####

df = pd.DataFrame(np.load('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dcoarsegrid.npy'))
# df = pd.DataFrame(sio.loadmat('/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/Programs/1DPDEs/Xdmd.mat')['Xdmd']).T
# df2 = pd.DataFrame(sio.loadmat('/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/Programs/1DPDEs/Phi.mat')['Phi'])
# df = np.dot(df1, df2.T)


# df_train = preprocessing.normalize(df[:int(0.7*df.shape[0])])
# df_eval = preprocessing.normalize(df[int(0.7*df.shape[0]):])
# dmd_train = preprocessing.normalize(dmd[:int(0.7*dmd.shape[0])])
# dmd_eval = preprocessing.normalize(dmd[int(0.7*dmd.shape[0]):])
# train_data = TensorDataset(torch.Tensor(np.concatenate([df_train, dmd_train], axis=1)))
# eval_data = TensorDataset(torch.Tensor(np.concatenate([df_eval, dmd_eval], axis=1)))



train_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[:int(0.7*df.shape[0])])))
eval_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[int(0.7*df.shape[0]):])))
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=shuffle)



##### NN #####

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
            
            # torch.nn.init.uniform_(self.encoder[0].weight.data, 0.0001, 0.1)
            # print("\nWeight after sampling from Uniform Distribution:\n")
            # print(self.encoder[0].weight)
 
        
        def forward(self, x):
            
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            
            return decoded
        
        
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs/2], gamma=0.1)
    
    
    train_loss = []
    eval_loss = []
    
    
    def evaluate(model, eval_loader):
        model.eval()
        running_eval_loss = 0
        
        for i, x in enumerate(eval_loader):
            input_eval = torch.Tensor(x[0][0])
            recone = model(input_eval)
            losse = criterion(recone, input_eval)
            running_eval_loss += losse.item()
        eval_loss.append(running_eval_loss / len(eval_loader))
        print(f"Evaluation Loss: {round(running_eval_loss / len(eval_loader), 20), latent_dim}")

        
    for epoch in range(epochs):
        
        #  Extract intial weights
        # if epoch == 0 and latent_dim == 1:
        #     for name, param in model.named_parameters():
        #         if name == 'encoder.0.weight':
        #             print(name, param[0])
        #             initial_weights = param[0]
        # break
        
        model.train()
        running_train_loss = 0
        print(f'Epoch: {epoch +1}')
        
        for i, y in enumerate(train_loader):
            input_train = torch.Tensor(y[0][0])
            recont = model(input_train)
            losst = criterion(recont, input_train)
            optimizer.zero_grad()
            losst.backward()
            optimizer.step()
            running_train_loss += losst.item()
        # scheduler.step()
        
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])
        train_loss.append(running_train_loss / len(train_loader))
        evaluate(model, eval_loader)
        print(f'Training Loss: {round(running_train_loss / len(train_loader), 20)}')
    evalDict[latent_dim].append(eval_loss[-1])
    
  

fails = 0  
for i in range(iterations):  # Compare _ number of iterations
    
    decreasing = True  # Evaluation losses decreasing 
    
    while decreasing and fails < 8:  # Enforce decreasing losses
        
        evalDict = {}  # Losses for each latent dimension
        
        for latent_dim in range(1, latent_dim+1):  # Dimension of manifold
            
            if latent_dim not in evalDict:
                evalDict[latent_dim] = []
                
            autoencoder()
            
        latent_dims = []
        losses = []
        
        for key, value in evalDict.items():
            
            latent_dims.append(key)
            losses.append(value[0])
        
                
        # Enforce decreasing losses
        for idx, loss in enumerate(losses):  
            
            if idx == 0 or idx == 1:  # Skip the first eval loss
                continue
            
            elif (loss / losses[idx-1]) < 10 and idx < 3:  # Continue if this error smaller than the last
                continue
            
            elif (loss / losses[idx-1]) > 10 and idx < 3:  # Stop and restart if this error bigger than the last
                fails += 1
                break
            
            elif idx == 3 and (loss / losses[idx-1]) < 10:  # Display results when the end is reached
                plt.plot(latent_dims, losses, marker='o', label=f'{i+1}')
                plt.legend() 
                decreasing = False  # Finished (eval losses done decreasing)
                break
            
            else: 
                fails += 1
                break


if fails == 0:  # Didn't need to correct losses
    print('CONSISTENT RESULTS')

else:
    print(f'{fails} TRIES')



##### Plot results #####

print(losses)
plt.xticks(latent_dims)
# plt.yticks(np.log([10e-4, 10e-5, 10e-6, 10e-7]), ['10e-4', '10e-5', '10e-6', '10e-7'])
plt.xlabel('Latent Space Dimension')
plt.ylabel('Evaluation Loss')
plt.title(f'A={a}, B={b}, t={t}, x={x}, k={k*dx} * 1/dx, q={q*dx} * 1/dx')
plt.show()
