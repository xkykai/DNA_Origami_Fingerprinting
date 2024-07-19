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
import pywt
torch.manual_seed(123)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
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

species = np.zeros((nrows, 2), bool)
for i in range(len(origami1_data)):
    species[i, 0] = 1

for i in range(len(origami2_data)):
    species[i+len(origami1_data), 1] = 1
#%%
times = np.arange(0, 40, 1)
scales = np.geomspace(1, 128, num=40)
wavelet = "cgau1"
coef, freqs = pywt.cwt(origami2_data[1000, :], scales, wavelet)
cwtmatr = np.abs(coef[:-1, :-1])

# plt.figure()
# plt.pcolormesh(times, freqs, cwtmatr)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.show()
#%%
times = np.arange(0, 40, 1)
scales = np.geomspace(1, 128, num=40)
nscales = len(scales) - 1
nfreqs = N - 1
wavelet = "cgau1"

data_cwt = np.zeros((nrows, 1, nfreqs, nscales))
for i in range(len(origami1_data)):
    coef, freqs = pywt.cwt(origami1_data[i, :], scales, wavelet)
    cwtmatr = np.abs(coef[:-1, :-1])
    data_cwt[i, 0, :, :] = cwtmatr

for i in range(len(origami2_data)):
    coef, freqs = pywt.cwt(origami2_data[i, :], scales, wavelet)
    cwtmatr = np.abs(coef[:-1, :-1])
    data_cwt[i+len(origami1_data), 0, :, :] = cwtmatr

#%%
# plt.figure()
# plt.pcolormesh(times, freqs, data_cwt[2000, 0, :, :])
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.show()
#%%
X = torch.tensor(data_cwt).float().to(device)
# X = (X - torch.mean(X, dim=0)) / (1 * torch.std(X, dim=0))
Y = torch.tensor(species).float().to(device)

X = torch.nan_to_num(X, nan=0)
X_train, X_not_train, Y_train, Y_not_train = train_test_split(X, Y, train_size=0.75, random_state=1)
X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, train_size=0.6, random_state=1)
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

batch_size = 64

trainloader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(ds_val,batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(ds_test,batch_size=batch_size, shuffle=True)

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
            nn.Conv2d(1, 8, 3),
            activation,
            nn.MaxPool2d(2, padding=1),
            nn.Conv2d(8, 16, 5),
            activation,
            nn.MaxPool2d(2, padding=1),
            activation,
            nn.Conv2d(16, 32, 5),
            activation,
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, hidden_layer_size),
            activation,
            *[nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), activation) for _ in range(N_hidden_layers-1)],
            nn.Linear(hidden_layer_size, output_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
hidden_layer_size = 128
N_hidden_layers = 2
activation = nn.GELU()
activation_str = str(activation).split("(")[0]
model = NN(len(ds_train[0][0]), 2, hidden_layer_size, N_hidden_layers, activation)

# for batch, (X, y) in enumerate(trainloader):
#     print(np.shape(model(X)))
#     break
#%%
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

epochs = 200

def train_model(hidden_layer_size, N_hidden_layers, activation):
    model = NN(len(ds_train[0][0]), 2, hidden_layer_size, N_hidden_layers, activation)
    activation_str = str(activation).split("(")[0]
    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    FILE_DIR = f"CWT_CNN_weights/{activation_str}_{hidden_layer_size}_{N_hidden_layer}_Adam"
    os.makedirs(FILE_DIR, exist_ok=True)

    min_loss = np.inf
    max_accuracy = 0

    min_loss_state = model.state_dict()
    max_accuracy_state = model.state_dict()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        train_loss[t], train_accuracy[t] = test_loop(trainloader, model, loss_fn)
        val_loss[t], val_accuracy[t] = test_loop(valloader, model, loss_fn)
        test_loss[t], test_accuracy[t] = test_loop(testloader, model, loss_fn)
        print(f"Train Error: \n Accuracy: {(train_accuracy[t]):>0.1f}%, Avg loss: {train_loss[t]:>8f} \n")
        print(f"Validation Error: \n Accuracy: {(val_accuracy[t]):>0.1f}%, Avg loss: {val_loss[t]:>8f} \n")
        print(f"Test Error: \n Accuracy: {(test_accuracy[t]):>0.1f}%, Avg loss: {test_loss[t]:>8f} \n")

        if val_loss[t] < min_loss:
            min_loss = val_loss[t]
            min_loss_state = model.state_dict()
            torch.save(model.state_dict(), f'{FILE_DIR}/model_weights_min_loss.pth')
        if val_accuracy[t] > max_accuracy:
            max_accuracy = val_accuracy[t]
            max_accuracy_state = model.state_dict()
            torch.save(model.state_dict(), f'{FILE_DIR}/model_weights_max_accuracy.pth')

    min_loss = np.amin(val_loss)
    epoch_min_loss = np.argmin(val_loss)
    max_accuracy = np.amax(val_accuracy)
    epoch_max_accuracy = np.argmax(val_accuracy)

    print("Done!")

    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label=f"Validation loss")
    plt.plot(test_loss, label="Test loss")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Min loss: {min_loss:.4f} at epoch {epoch_min_loss}")
    plt.xlabel("Epochs")
    plt.ylabel("Cross entropy loss")
    plt.legend()
    plt.savefig(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_loss.png", dpi=1000)

    plt.figure()
    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(val_accuracy, label=f"Validation accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Max accuracy: {max_accuracy:.2f}% at epoch {epoch_max_accuracy}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_accuracy.png", dpi=1000)

    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_loss.csv", train_loss, delimiter=",")
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_loss.csv", val_loss, delimiter=",")
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_loss.csv", test_loss, delimiter=",")

    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_accuracy.csv", train_accuracy, delimiter=",")
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_accuracy.csv", val_accuracy, delimiter=",")
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_accuracy.csv", test_accuracy, delimiter=",")

    model.load_state_dict(torch.load(f'{FILE_DIR}/model_weights_max_accuracy.pth'))
    model.eval()

    y_pred_train = []
    y_true_train = []

    # iterate over test data
    for inputs, labels in trainloader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred_train.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true_train.extend(labels) # Save Truth

    y_true_train = [np.argmax(y) for y in y_true_train]

    # constant for classes
    classes = ("Species 1", "Species 2")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_train, y_pred_train)
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Training")
    plt.savefig(f'./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_training_confusion.png', dpi=1000)

    y_pred_val = []
    y_true_val = []

    # iterate over test data
    for inputs, labels in valloader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred_val.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true_val.extend(labels) # Save Truth


    y_true_val = [np.argmax(y) for y in y_true_val]

    # constant for classes
    classes = ("Species 1", "Species 2")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_val, y_pred_val)
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Validation")
    plt.savefig(f'./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_validation_confusion.png', dpi=1000)

    y_pred_test = []
    y_true_test = []

    # iterate over test data
    for inputs, labels in testloader:
            output = model(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred_test.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true_test.extend(labels) # Save Truth


    y_true_test = [np.argmax(y) for y in y_true_test]

    # constant for classes
    classes = ("Species 1", "Species 2")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true_test, y_pred_test)
    np.savetxt(f"./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Testing")
    plt.savefig(f'./Output/CWT_CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_confusion.png', dpi=1000)
#%%
# hidden_layer_sizes = [64, 128, 256]
# N_hidden_layers = [1, 2]
hidden_layer_sizes = [63]
N_hidden_layers = [2]
# activation = [nn.GELU(), nn.ReLU(), nn.LeakyReLU(), nn.SiLU()]
# activation = [nn.GELU()]
# activation = [nn.ReLU()]
# activation = [nn.LeakyReLU()]
activation = [nn.SiLU()]

for hidden_layer_size in hidden_layer_sizes:
    for N_hidden_layer in N_hidden_layers:
        for act in activation:
            print(f"Training model with hidden layer size {hidden_layer_size}, {N_hidden_layer} hidden layers, and activation function {str(act).split('(')[0]}")
            train_model(hidden_layer_size, N_hidden_layer, act)
#%%
