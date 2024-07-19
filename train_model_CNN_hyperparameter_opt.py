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
X = torch.tensor(np.concatenate((origami1_data, origami2_data))).float()
X = (X - torch.mean(X, dim=0)) / torch.std(X, dim=0)
Y = torch.tensor(species).float()

X = torch.nan_to_num(X, nan=0)
X = torch.reshape(X, (X.shape[0], 1, X.shape[1]))
X_train, X_not_train, Y_train, Y_not_train = train_test_split(X, Y, train_size=0.75, random_state=100)
X_val, X_test, Y_val, Y_test = train_test_split(X_not_train, Y_not_train, train_size=0.6, random_state=100)
#%%
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
            # nn.Linear(input_dim, hidden_layer_size),
            # activation,
            # *[nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), activation) for _ in range(N_hidden_layers)],
            # nn.Linear(hidden_layer_size, output_dim),
            # nn.Softmax()

            nn.Conv1d(1, 8, 3, padding=1),
            activation,
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5),
            activation,
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            activation,
            nn.Flatten(),
            nn.Linear(128, hidden_layer_size),
            *[nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), activation) for _ in range(N_hidden_layers-1)],
            nn.Linear(hidden_layer_size, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

hidden_layer_size = 128
N_hidden_layers = 2
activation = nn.GELU()
activation_str = str(activation).split("(")[0]
model = NN(len(ds_train[0][0]), 2, hidden_layer_size, N_hidden_layers, activation)
# model(ds_train[0][0])
# model(ds_train[0])
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

epochs = 300

def train_model(hidden_layer_size, N_hidden_layers, activation):
    model = NN(len(ds_train[0][0]), 2, hidden_layer_size, N_hidden_layers, activation)
    activation_str = str(activation).split("(")[0]
    learning_rate = 3e-4
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    WEIGHTS_DIR = f"./CNN_weights/{activation_str}_{hidden_layer_size}_{N_hidden_layer}_Adam"
    OUTPUT_DIR = f"./Output/CNN"
    DATA_DIR = f"./Data/CNN"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

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
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_weights_min_loss.pth')
        if val_accuracy[t] > max_accuracy:
            max_accuracy = val_accuracy[t]
            max_accuracy_state = model.state_dict()
            torch.save(model.state_dict(), f'{WEIGHTS_DIR}/model_weights_max_accuracy.pth')

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
    plt.savefig(f"{OUTPUT_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_loss.png", dpi=1000)

    plt.figure()
    plt.plot(train_accuracy, label="Training accuracy")
    plt.plot(val_accuracy, label=f"Validation accuracy")
    plt.plot(test_accuracy, label="Test accuracy")
    plt.title(f"Hidden layer size: {hidden_layer_size}, {N_hidden_layers} hidden layers, activation function: {activation_str} \n Max accuracy: {max_accuracy:.2f}% at epoch {epoch_max_accuracy}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_accuracy.png", dpi=1000)

    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_loss.csv", train_loss, delimiter=",")
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_loss.csv", val_loss, delimiter=",")
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_loss.csv", test_loss, delimiter=",")

    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_accuracy.csv", train_accuracy, delimiter=",")
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_accuracy.csv", val_accuracy, delimiter=",")
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_accuracy.csv", test_accuracy, delimiter=",")

    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/model_weights_max_accuracy.pth'))
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
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_train_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Training")
    plt.savefig(f'{OUTPUT_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_training_confusion.png', dpi=1000)

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
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_val_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Validation")
    plt.savefig(f'{OUTPUT_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_validation_confusion.png', dpi=1000)

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
    np.savetxt(f"{DATA_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_confusion.csv", cf_matrix, delimiter=",")
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure()
    ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
    ax.set_title("Testing")
    plt.savefig(f'{OUTPUT_DIR}/CNN_{activation_str}_hidden_{hidden_layer_size}_layer_{N_hidden_layers}_Adam_test_confusion.png', dpi=1000)
#%%
hidden_layer_sizes = [64, 128, 256]
N_hidden_layers = [1, 2, 3, 4]
# hidden_layer_sizes = [255]
# N_hidden_layers = [2]
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
