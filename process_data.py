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
    def __init__(self, input_dim, output_dim):
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
            nn.Linear(input_dim, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, output_dim),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

model = NN(len(ds_train[0][0]), 2)
model(ds_train[0][0])
#%%
learning_rate = 2e-3
batch_size = 256
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
#%%
epochs = 2000
train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
test_loss = np.zeros(epochs)

train_accuracy = np.zeros(epochs)
val_accuracy = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainloader, model, loss_fn, optimizer)
    train_loss[t], train_accuracy[t] = test_loop(trainloader, model, loss_fn)
    val_loss[t], val_accuracy[t] = test_loop(valloader, model, loss_fn)
    test_loss[t], test_accuracy[t] = test_loop(testloader, model, loss_fn)
    print(f"Train Error: \n Accuracy: {(train_accuracy[t]):>0.1f}%, Avg loss: {train_loss[t]:>8f} \n")
    print(f"Validation Error: \n Accuracy: {(val_accuracy[t]):>0.1f}%, Avg loss: {val_loss[t]:>8f} \n")
    print(f"Test Error: \n Accuracy: {(test_accuracy[t]):>0.1f}%, Avg loss: {test_loss[t]:>8f} \n")
    torch.save(model.state_dict(), f'NN_weights/GELU_hidden_100_layer_2/model_weights_{t}.pth')

print("Done!")
#%%
# model.load_state_dict(torch.load("NN_weights\GELU_hidden_100_layer_2\model_weights_915.pth"))
# model.eval()
#%%
plt.plot(train_loss, label="Training loss")
plt.plot(val_loss, label="Validation loss")
plt.plot(test_loss, label="Test loss")
plt.xlabel("Epochs")
plt.ylabel("Cross entropy loss")
plt.legend()
plt.savefig("GELU_hidden_100_layer_2_loss.png", dpi=1000)
plt.show()

#%%
plt.plot(train_accuracy, label="Training accuracy")
plt.plot(val_accuracy, label="Validation accuracy")
plt.plot(test_accuracy, label="Test accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("GELU_hidden_100_layer_2_accuracy.png", dpi=1000)
plt.show()
#%%
np.argmin(val_loss)
np.amin(val_loss)
np.amax(val_accuracy)
test_loss[np.argmin(val_loss)]
test_accuracy[np.argmin(val_loss)]

#%%
row_n = 50
y = origami1_data[row_n,:]
N = len(y)
T = 1 / 50
x = np.linspace(0, N*T, N)
xf = sp.fft.fftfreq(N, T)[:N//2]
yf = sp.fft.fft(y)

xf_inv = sp.fft.ifft(yf)
#%%
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()
#%%
plt.plot(origami1_data[1,:])
plt.plot(origami1_data[50,:])
plt.show()
#%%
T = 1 / 50
xf = sp.fft.fftfreq(N, T)[:N//2]

for i in range(100):
    plt.plot(data_fft[i, 0:N//2], linewidth=1)

plt.xlabel(r"FFT frequency (s$^{-1}$)")
plt.ylabel("Fourier amplitude")
plt.title("Species 1")
plt.savefig("FFT_1.png", dpi=1000)
plt.show()

for i in range(100):
    plt.plot(data_fft[i+len(origami1_data), 0:N//2], linewidth=1)

plt.xlabel(r"FFT frequency (s$^{-1}$)")
plt.ylabel("Fourier amplitude")
plt.title("Species 2")
plt.savefig("FFT_2.png", dpi=1000)
plt.show()

#%%
model.load_state_dict(torch.load("NN_weights\GELU_hidden_100_layer_2\model_weights_1016.pth"))
model.eval()
#%%
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure()
ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
ax.set_title("Training")
plt.savefig('training_confusion.png', dpi=1000)
#%%
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure()
ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
ax.set_title("Validation")
plt.savefig('validation_confusion.png', dpi=1000)
#%%
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
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure()
ax = sn.heatmap(df_cm, annot=True, cbar_kws={"label":"Probability"}, square=True)
ax.set_title("Testing")
plt.savefig('test_confusion.png', dpi=1000)
#%%