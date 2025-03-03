import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from DeepCCAModels import DeepCCA
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# Load datasets from CSV files
X_train = pd.read_csv('X_train.csv', header=None).values
Y_train = pd.read_csv('Y_train.csv', header=None).values
X_test = pd.read_csv('X_test.csv', header=None).values
Y_test = pd.read_csv('Y_test.csv', header=None).values


# Verify the shapes of the loaded data
print(f'X_train shape: {X_train.shape}')
print(f'Y_train shape: {Y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'Y_test shape: {Y_test.shape}')

# Ensure the number of samples in X and Y are the same
assert X_train.shape[0] == Y_train.shape[0], "Mismatch in number of samples between X_train and Y_train"
assert X_test.shape[0] == Y_test.shape[0], "Mismatch in number of samples between X_test and Y_test"

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.double)
Y_train = torch.tensor(Y_train, dtype=torch.double)
X_test = torch.tensor(X_test, dtype=torch.double)
Y_test = torch.tensor(Y_test, dtype=torch.double)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

# Define the DCCA model
input_dim_X = X_train.shape[1]  # Number of features in X
input_dim_Y = 2     # Number of features in Y
hidden_dim = 1000
output_dim = 2
layer_sizes_X = [hidden_dim, output_dim]
layer_sizes_Y = [hidden_dim, output_dim]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepCCA(layer_sizes_X, layer_sizes_Y, input_dim_X, input_dim_Y, output_dim, use_all_singular_values=False, device=device).to(device)

# Training the model
optimizer = optim.LBFGS(list(model.parameters()), lr=0.1)

def train(epoch):
    model.train()
    for data_X, data_Y in train_loader:
        data_X = data_X.to(device)
        data_Y = data_Y.to(device)

        def closure():
            optimizer.zero_grad()
            output_X, output_Y = model(data_X, data_Y)
            loss = model.loss(output_X, output_Y)
            loss.backward()
            return loss

        optimizer.step(closure)

for epoch in range(1, 11):
    train(epoch)
    print(f'Epoch {epoch} complete')

# Evaluate the model
model.eval()
dcca_corrs = np.zeros(output_dim)
with torch.no_grad():
    for data_X, data_Y in test_loader:
        data_X = data_X.to(device)
        data_Y = data_Y.to(device)
        output_X, output_Y = model(data_X, data_Y)
        for i in range(1, output_dim + 1):
            corr = np.corrcoef(output_X[:, :i].cpu().numpy().T, output_Y[:, :i].cpu().numpy().T).diagonal(offset=i).mean()
            if np.isnan(corr) or np.isinf(corr):
                corr = 0  # Handle NaN or infinite values
                print("-")
            dcca_corrs[i-1] += corr
dcca_corrs /= len(test_loader)
print(f'DCCA Test correlation: {np.mean(dcca_corrs)}')

# Linear CCA
cca = CCA(n_components=output_dim)
cca.fit(X_train.numpy(), Y_train.numpy())
cca_X, cca_Y = cca.transform(X_test.numpy(), Y_test.numpy())
cca_corrs = [np.corrcoef(cca_X[:, :i].T, cca_Y[:, :i].T).diagonal(offset=i).mean() for i in range(1, output_dim + 1)]
print(f'Linear CCA Test correlation: {cca_corrs[-1]}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, output_dim + 1), dcca_corrs, label='DCCA')
plt.plot(range(1, output_dim + 1), cca_corrs, label='Linear CCA')
plt.xlabel('Number of Components')
plt.ylabel('Correlation')
plt.title('Correlation as a Function of Number of Components')
plt.legend()
plt.grid(True)
plt.show()
