import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from DeepCCAModels import DeepCCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# Split dataset into left and right halves
def split_mnist(dataset):
    left_data = []
    right_data = []
    for img, label in dataset:
        left_data.append(img[:, :, :14].reshape(-1))
        right_data.append(img[:, :, 14:].reshape(-1))
    return torch.stack(left_data), torch.stack(right_data)


train_left, train_right = split_mnist(train_dataset)
test_left, test_right = split_mnist(test_dataset)

# Convert data to double precision
train_left = train_left.double()
train_right = train_right.double()
test_left = test_left.double()
test_right = test_right.double()

# Create DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(train_left, train_right), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_left, test_right), batch_size=batch_size, shuffle=False)

# Define the DCCA model
input_dim = 14 * 28
hidden_dim = 1000
output_dim = 50
layer_sizes1 = [hidden_dim, output_dim]
layer_sizes2 = [hidden_dim, output_dim]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepCCA(layer_sizes1, layer_sizes2, input_dim, input_dim, output_dim, use_all_singular_values=False,
                device=device).to(device)

# Training the model
optimizer = optim.LBFGS(list(model.parameters()), lr=0.1)


def train(epoch):
    model.train()
    for data_left, data_right in train_loader:
        data_left = data_left.to(device)
        data_right = data_right.to(device)

        def closure():
            optimizer.zero_grad()
            output_left, output_right = model(data_left, data_right)
            loss = model.loss(output_left, output_right)
            loss.backward()
            return loss

        optimizer.step(closure)


for epoch in range(1, 11):
    train(epoch)
    print(f'Epoch {epoch} complete')

# Evaluate the model
model.eval()
dcca_corrs = []

with torch.no_grad():
    for data_left, data_right in test_loader:
        data_left = data_left.to(device)
        data_right = data_right.to(device)

        output_left, output_right = model(data_left, data_right)

        # Compute batch correlations manually
        H1, H2 = output_left.T, output_right.T  # Transpose to [features x batch_size]
        SigmaHat12 = torch.matmul(H1, H2.T) / (H1.shape[1] - 1)  # Cross-covariance
        SigmaHat11 = torch.matmul(H1, H1.T) / (H1.shape[1] - 1) + 1e-3 * torch.eye(output_dim, device=device)
        SigmaHat22 = torch.matmul(H2, H2.T) / (H2.shape[1] - 1) + 1e-3 * torch.eye(output_dim, device=device)

        # Whitening step
        D1, V1 = torch.linalg.eigh(SigmaHat11)
        D2, V2 = torch.linalg.eigh(SigmaHat22)
        D1, V1 = D1[D1 > 1e-9], V1[:, D1 > 1e-9]
        D2, V2 = D2[D2 > 1e-9], V2[:, D2 > 1e-9]

        SigmaHat11RootInv = V1 @ torch.diag(D1 ** -0.5) @ V1.T
        SigmaHat22RootInv = V2 @ torch.diag(D2 ** -0.5) @ V2.T

        # Canonical correlations
        Tval = SigmaHat11RootInv @ SigmaHat12 @ SigmaHat22RootInv
        singular_values = torch.linalg.svdvals(Tval)  # Extract all correlations

        dcca_corrs.append(singular_values.cpu().numpy())  # Store per-batch correlations

# Average across batches
dcca_corrs = np.mean(np.array(dcca_corrs), axis=0)
print(f'DCCA Test correlation: {np.sum(dcca_corrs)}')

# Linear CCA
cca = CCA(n_components=output_dim)
cca.fit(train_left.numpy(), train_right.numpy())
cca_left, cca_right = cca.transform(test_left.numpy(), test_right.numpy())

# Compute correlation coefficients for each component
cca_corrs = np.array([np.corrcoef(cca_left[:, i], cca_right[:, i])[0, 1] for i in range(output_dim)])

print(f'Linear CCA Test correlation: {np.sum(cca_corrs)}')

# Kernel CCA (KCCA) with RBF kernel
# gamma = 1.0 / (2 * (14 * 28) ** 2)
# kcca = CCA(n_components=output_dim)
# kpca_left = KernelPCA(kernel="rbf", gamma=gamma, fit_inverse_transform=True)
# kpca_right = KernelPCA(kernel="rbf", gamma=gamma, fit_inverse_transform=True)
# train_left_kpca = kpca_left.fit_transform(train_left.numpy())
# train_right_kpca = kpca_right.fit_transform(train_right.numpy())
# kcca.fit(train_left_kpca, train_right_kpca)
# test_left_kpca = kpca_left.transform(test_left.numpy())
# test_right_kpca = kpca_right.transform(test_right.numpy())
# kcca_left, kcca_right = kcca.transform(test_left_kpca, test_right_kpca)
# kcca_corrs = [np.corrcoef(kcca_left[:, :i].T, kcca_right[:, :i].T).diagonal(offset=i).mean() for i in range(1, output_dim + 1)]
# print(f'Kernel CCA (RBF) Test correlation: {kcca_corrs[-1]}')

dcca_cumulative = [np.sum(dcca_corrs[:i]) for i in range(1, output_dim + 1)]
cca_cumulative = [np.sum(cca_corrs[:i]) for i in range(1, output_dim + 1)]
# kcca_cumulative = [np.sum(kcca_corrs[:i]) for i in range(1, output_dim + 1)]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, output_dim + 1), dcca_cumulative, label='DCCA')
plt.plot(range(1, output_dim + 1), cca_cumulative, label='Linear CCA')
# plt.plot(range(1, output_dim + 1), kcca_corrs, label='Kernel CCA (RBF)')
plt.xlabel('Number of Components')
plt.ylabel('Correlation')
plt.title('Correlation as a Function of Number of Components')
plt.legend()
plt.grid(True)
plt.show()
