import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA, PCA
import numpy as np

##### Parameters #####

train_split = 3500  # How many rows for training
input_size = 64
batch_size = 30
shuffle = True 


def autoencoder():
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, latent_dim, bias=True, dtype=torch.float32))

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 500, bias=True, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(500, input_size, bias=True, dtype=torch.float32))

        def forward(self, x):
            phase_x = torch.Tensor(x[0][0][:input_size])  # 1x64
            pca_x = torch.Tensor(x[0][0][input_size:(input_size + latent_dim)])  # 1xlatent_dim
            inv_pca_x = torch.Tensor(x[0][0][(input_size + latent_dim):])  # 1x64
            encoded = self.encoder(phase_x)  # 1xlatent_dim
            encoded += pca_x  # 1xlatent_dim
            decoded = self.decoder(encoded) + inv_pca_x
            return decoded

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)

    num_epochs = 400
    train_loss = []
    eval_loss = []
    # lr1
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50, 60, 70, 80], gamma=0.1)
    
    #lr2
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    # batch size updates weights based of average of data in batch

    def evaluate(model, validation_loader):
        model.eval()
        running_loss = 0
        for i1, x1 in enumerate(validation_loader):
            recon1 = model(x1)
            loss_eval = criterion(x1[0][0][:input_size], recon1)
            running_loss += loss_eval.item()
        eval_loss.append(running_loss / len(validation_loader))
        print(f"Evaluation loss: {round(running_loss / len(validation_loader), 20), latent_dim}")
        # len(validation_loader) is number of batches in data

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        print(f'Epoch:{epoch + 1}')
        for i, x in enumerate(training_loader):
            # print(x[0][0][:64])
            recon = model(x)
            loss = criterion(x[0][0][:input_size], recon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('lr:', param_group['lr'])
        print(f"Training loss: {round(running_train_loss / len(training_loader), 20)}")
        train_loss.append(running_train_loss)
        evaluate(model, validation_loader)

    evalDict[a].append(eval_loss[-1])


def autoencoder2():
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, latent_dim, bias=True, dtype=torch.float32))

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 500, bias=True, dtype=torch.float32),
                nn.Softmax(),
                nn.Linear(500, input_size, bias=True, dtype=torch.float32))

        def forward(self, x):
            phase_x = torch.Tensor(x[0][0][:input_size])  # 1x64
            pca_x = torch.Tensor(x[0][0][input_size:(input_size + latent_dim)])  # 1xlatent_dim
            inv_pca_x = torch.Tensor(x[0][0][(input_size + latent_dim):])  # 1x64
            encoded = self.encoder(phase_x)  # 1xlatent_dim
            encoded -= pca_x  # 1xlatent_dim
            decoded = self.decoder(encoded)
            return decoded

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)

    num_epochs = 400
    train_loss = []
    eval_loss = []
    # lr1
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 50, 60, 70, 80], gamma=0.1)
    
    #lr2
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    # batch size updates weights based of average of data in batch

    def evaluate(model, validation_loader):
        model.eval()
        running_loss = 0
        for i1, x1 in enumerate(validation_loader):
            recon1 = model(x1)
            loss_eval = criterion(x1[0][0][:input_size], recon1)
            running_loss += loss_eval.item()
        eval_loss.append(running_loss / len(validation_loader))
        print(f"Evaluation loss: {round(running_loss / len(validation_loader), 20), latent_dim}")
        # len(validation_loader) is number of batches in data

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        print(f'Epoch:{epoch + 1}')
        for i, x in enumerate(training_loader):
            # print(x[0][0][:64])
            recon = model(x)
            loss = criterion(x[0][0][:input_size], recon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('lr:', param_group['lr'])
        print(f"Training loss: {round(running_train_loss / len(training_loader), 20)}")
        train_loss.append(running_train_loss)
        evaluate(model, validation_loader)

    evalDict[3].append(eval_loss[-1])

evalDict = {0: [], 1: [], 2: [], 3: []}  # 0: L=22, 1: L=44, 2: L=66
labelsDict = {0: 22, 1: 44, 2: 66}
manifoldDimDict = {0: 8, 1: 18, 2: 28}


# Run autoencoder
for a in range(0, 1):  # Picking KS system (L=22, 44, or 66) from labelsDict

    system = labelsDict[a]

    # Load Data
    df = pd.read_csv(
        "/Users/darinmomayezi/Documents/Research/GrigorievLab/Autoencoder/Data/KS_L={}_tf=10000_dt=.25_D=64.csv".format(
            system))

    ranging = []
    
    for i in range(manifoldDimDict[a] - 3, manifoldDimDict[a] + 3):  # Picking latent space dimension
        latent_dim = i
        ranging.append(latent_dim)
        
        # PCA
        pca = PCA(n_components=latent_dim)
        princ_comps = pca.fit_transform(df)
        inv_comps = pca.inverse_transform(princ_comps)
        
        # Data split
        pca_training = preprocessing.normalize(princ_comps[:train_split])  #
        pca_validation = preprocessing.normalize(princ_comps[train_split:])  #
        inv_pca_training = preprocessing.normalize(inv_comps[:train_split])  #
        inv_pca_validation = preprocessing.normalize(inv_comps[train_split:])  #
        training_data = preprocessing.normalize(df[:train_split])  #
        validation_data = preprocessing.normalize(df[train_split:])  #
        training = TensorDataset(
            torch.Tensor(np.concatenate([training_data, pca_training, inv_pca_training], axis=1)))
        validation = TensorDataset(
            torch.Tensor(np.concatenate([validation_data, pca_validation, inv_pca_validation], axis=1)))
        training_loader = torch.utils.data.DataLoader(dataset=training, batch_size=batch_size, shuffle=shuffle)
        validation_loader = torch.utils.data.DataLoader(dataset=validation, batch_size=batch_size, shuffle=shuffle)

        # if system == 22 or system == 44:
        #     autoencoder()
        # elif system == 66:
        #     autoencoder2()
        # autoencoder()
        autoencoder()

# Graph
# ranging = [-3, -2, -1, 0, 1, 2]
print(evalDict)
# plt.plot(ranging, np.log(evalDict[0]), label='L=22, ReLU', marker='o', color='b')
plt.plot(ranging, np.log(evalDict[3]), label='L=22, Sigmoid', marker='^', color='orange')
#plt.plot(ranging, np.log(evalDict[1]), label='L=44', marker='s', color='tab:orange')
#plt.plot(ranging, np.log(evalDict[2]), label='L=66', marker='v', color='g')
plt.yticks(np.log([1e-6, 1e-5, 1e-4, 1e-3]), ['10^-6', '10^-5', '10^-4', '10^-3'])
plt.legend()
plt.ylabel('Evaluation Loss')
# plt.xlabel('H - M')
plt.xlabel('d_H')
plt.title('Kernel PCA, a=1, b=1')
plt.show()
