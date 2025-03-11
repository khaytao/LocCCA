from DeepCCAModels import DeepCCA


DATA_PATH_X = "X_test.csv"
DATA_PATH_Y = "Y_test.csv"

model = DeepCCA()

model.load_state_dict(torch.load("model_state.pth"))

model.eval()


X = np.loadtxt(DATA_PATH_X, delimiter=',')
Y = np.loadtxt(DATA_PATH_Y, delimiter=',')

X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()


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