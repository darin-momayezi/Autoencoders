import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from OneDmeshgrid import x, dx, dt, k, q, a, b
import scipy.io as sio



##### Data #####

df = pd.DataFrame(np.load('/Users/darinmomayezi/Downloads/NSE2D[71]/1Dmeshgrid.npy'))
# df = pd.DataFrame(sio.loadmat('/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/Programs/1DPDEs/Xdmd.mat')['Xdmd']).T
# df = df[1:]
# print(df)



##### Parameters #####

input_size = 2*x  # Number of spatial grid points from OneDcoarsegrid
hidden1 = int(input_size/2)  # Size of hidden layer
hidden2 = int(input_size/10)
latent_dim = 4
epochs = 10000
lr = 0.01  # Learning rate
batch_size = 30
iterations = 3  # Run autoencoder _ times
train_split = 0.8  # Fraction of data for training
shuffle = True
    
# Weight decay
if a == 0 and b == 1:
    weight_decay = 0
elif a == 1 and b == 0:
    weight_decay = 0
else: weight_decay = 0



train_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[:int(train_split*df.shape[0])])))
eval_data = TensorDataset(torch.Tensor(preprocessing.normalize(df[int(train_split*df.shape[0]):])))
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=shuffle)



##### NN #####

def autoencoder():
    
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            
            self.encoder = nn.Sequential(nn.Linear(input_size, hidden2, bias=True, dtype=torch.float32),
                                         nn.Sigmoid(),
                                         nn.Linear(hidden2, latent_dim, bias=True, dtype=torch.float32))
            
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden2, bias=True, dtype=torch.float32),
                                         nn.Sigmoid(),
                                         nn.Linear(hidden2, input_size, bias=True, dtype=torch.float32))
            
            torch.nn.init.uniform_(self.encoder[0].weight.data, -0.0001, 0.0001)
            # print("\nWeight after sampling from Uniform Distribution:\n")
            # print(self.encoder[0].weight)
 
        
        def forward(self, x):
            
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            
            return decoded
        
        
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    if a == 1 and b == 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100],gamma=0.1)    
    else:
        pass
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000],gamma=0.1)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150],gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=850)
    
    
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
        final_eval_loss = running_eval_loss / len(eval_loader)
        eval_loss.append(final_eval_loss)
        print(f"Evaluation Loss: {round(final_eval_loss, 20), latent_dim}")
        
    # Training loop
    for epoch in range(epochs):
        
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
        scheduler.step(losst)
        
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])
        train_loss.append(running_train_loss / len(train_loader))
        evaluate(model, eval_loader)
        print(f'Training Loss: {round(running_train_loss / len(train_loader), 20)}, Iteration {iteration+1}')
    evalDict[latent_dim].append(eval_loss[-1])
    
  

'''To enforce reproducibility quit if evaluation losses increase by more than 10 (defined as a fail) for 8 iterations.'''
fails = 0  
all_losses = []

for iteration in range(iterations):  # Compare _ number of iterations
    
    decreasing = True  # Evaluation losses decreasing (condition to continue autoencoder)
    
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
            all_losses.append(value[0])
        
                
        # Enforce decreasing losses
        for idx, loss in enumerate(losses):  
            
            if idx == 0 or idx == 1:  # Skip the first eval loss
                continue
            
            # elif loss > 8e-5:  # Get rid of losses that mess up graph
            #     break
            
            elif (loss / losses[idx-1]) < 10 and idx < 3:  # Continue if this error smaller than the last
                continue
            
            elif (loss / losses[idx-1]) > 10 and idx < 3:  # Stop and restart if this error bigger than the last
                fails += 1
                break
            
            elif idx == 3 and (loss / losses[idx-1]) < 10:  # Display results when the end is reached
                plt.plot(latent_dims, losses, marker='o', label=f'{iteration+1}')
                plt.legend(loc='upper right') 
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

print(all_losses)

print(losses)
plt.xticks(latent_dims)

# if a == 1 and b == 1:
#     # plt.yticks([5e-7, 1e-7, 1e-8], ['5*10^-7', '10^-7', '10^-8'])
#     pass
# elif a == 0 and b == 1:
#     plt.yticks([1e-9, 1e-10], ['10^-9', '10^-10'])
# elif a == 1 and b == 0:
#     plt.yticks([1e-6, 1e-7], ['10^-6', '10^-7'])
    
plt.xlabel('Latent Space Dimension')
plt.ylabel('Evaluation Loss')
plt.title(f'A={a}, B={b}, dt={dt}, dx={dx}, k={k*dx} * 1/dx, q={q*dx} * 1/dx, lr={lr}')
plt.show()
