#%%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
#%%
# df = pd.read_csv("Data/DNA/1-4-2019 dna 200mV 0kPa.txt", delimiter="\t", header=0)
df_1 = pd.read_excel("Data/origami1/output_2500_events.xlsx", header=None)
df_1 = df_1.dropna(how="all")
origami1_data = np.array(df_1)

df_2 = pd.read_excel("Data/origami2/output_2_2500_events.xlsx", header=None)
df_2 = df_2.dropna(how="all")
origami2_data = np.array(df_2)

for i in range(len(origami1_data)):
    origami1_data[i,:] = np.nan_to_num(origami1_data[i, :], nan=origami1_data[i,1])

for i in range(len(origami2_data)):
    origami2_data[i,:] = np.nan_to_num(origami2_data[i, :], nan=origami2_data[i,1])
#%%
nrows = len(origami1_data)+len(origami2_data)
N = np.amax([np.shape(origami1_data)[1], np.shape(origami2_data)[1]])
ncols = (N // 2) * 2

species = np.zeros((nrows, 2), bool)
for i in range(len(origami1_data)):
    species[i, 0] = 1

for i in range(len(origami2_data)):
    species[i+len(origami1_data), 1] = 1
#%%
data_fft = np.zeros((nrows, ncols))
for i in range(len(origami1_data)):
    yf = sp.fft.fft(origami1_data[i,:])[0:N//2]
    for j in range(len(yf)):
        data_fft[i, j] = np.real(yf)[j]
        data_fft[i, j+N//2] = np.imag(yf)[j]
    
for i in range(len(origami2_data)):
    yf = sp.fft.fft(origami2_data[i,:])[0:N//2]
    for j in range(len(yf)):
        data_fft[i+len(origami1_data), j] = np.real(yf)[j]
        data_fft[i+len(origami1_data), j+N//2] = np.imag(yf)[j]

#%%
X = torch.tensor(data_fft).float()
X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)
Y = torch.tensor(species).float()

X = torch.nan_to_num(X, nan=0)
X_train, X_not_train, Y_train, Y_not_train = train_test_split(X, Y, train_size=0.75)
X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, train_size=0.6)
#%%
class FFTData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        sample = {
            'feature': torch.tensor([self.x[index, :]], dtype=torch.float64), 
                  'label': torch.tensor([self.y[index, :]], dtype=torch.bool)}
        return sample

    def __len__(self):
        return len(self.x)

ds_train = torch.utils.data.TensorDataset(X_train, Y_train)
ds_val = torch.utils.data.TensorDataset(X_val, Y_val)
ds_test = torch.utils.data.TensorDataset(X_test, Y_test)

trainloader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(ds_val,batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test,batch_size=64, shuffle=True)

#%%
class NN(nn.Module):
    '''
    Class for fully connected neural net.
    '''
    def __init__(self, input_dim, output_dim, hidden_layer_size=100, N_hidden_layers=2, activation=nn.GELU()):
        '''
        Parameters
        ----------
        input_dim: int
            input dimension (i.e., # of features in each example passed to the network)
        hidden_dim: int
            number of nodes in hidden layer
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            # TODO: fill in network layers
            # Network should have a single hidden layer
            # Apply ReLU activation in between the hidden layer and output node
            nn.Linear(input_dim, hidden_layer_size),
            activation,
            *[nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), activation) for _ in range(N_hidden_layers)],
            nn.Linear(hidden_layer_size, output_dim),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # print(pred)
            # print(pred.argmax(-1))
            # print(len(y.argmax(-1)))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(-1) == y.argmax(-1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, 100*correct

epochs = 2000

def train_model(hidden_layer_size, N_hidden_layers, activation):
    model = NN(len(ds_train[0][0]), 2, hidden_layer_size, N_hidden_layers, activation)
    activation_str = str(activation).split("(")[0]
    learning_rate = 2e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    FILE_DIR = f"NN_weights/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}"
    os.makedirs(FILE_DIR, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        train_loss[t], train_accuracy[t] = test_loop(trainloader, model, loss_fn)
        val_loss[t], val_accuracy[t] = test_loop(valloader, model, loss_fn)
        test_loss[t], test_accuracy[t] = test_loop(testloader, model, loss_fn)
        print(f"Train Error: \n Accuracy: {(train_accuracy[t]):>0.1f}%, Avg loss: {train_loss[t]:>8f} \n")
        print(f"Validation Error: \n Accuracy: {(val_accuracy[t]):>0.1f}%, Avg loss: {val_loss[t]:>8f} \n")
        print(f"Test Error: \n Accuracy: {(test_accuracy[t]):>0.1f}%, Avg loss: {test_loss[t]:>8f} \n")
        torch.save(model.state_dict(), f'{FILE_DIR}/model_weights_{t}.pth')

    epoch_min_loss = np.argmin(val_loss)
    saved_files = os.listdir(FILE_DIR)
    except_file = f'model_weights_{epoch_min_loss}.pth'

    for filename in saved_files:
        if filename != except_file:
            os.remove(f'{FILE_DIR}/{filename}')

    print("Done!")

    plt.figure()
    plt.plot(val_loss, label="Validation loss")
    plt.plot(test_loss, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig(f"./Output/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_loss.png", dpi=1000)

    plt.figure()
    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(val_accuracy, label="Validation accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./Output/{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_accuracy.png", dpi=1000)
#%%
hidden_layer_sizes = [64, 128, 256]
N_hidden_layers = [2, 3]
activation = [nn.GELU(), nn.ReLU(), nn.LeakyReLU(), nn.SiLU()]

for hidden_layer_size in hidden_layer_sizes:
    for N_hidden_layer in N_hidden_layers:
        for act in activation:
            print(f"Training model with hidden layer size {hidden_layer_size}, {N_hidden_layer} hidden layers, and activation function {str(act).split('(')[0]}")
            train_model(hidden_layer_size, N_hidden_layer, act)

# %%
