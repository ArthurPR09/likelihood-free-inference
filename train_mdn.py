import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split


def mdn_loss(model, y, pi, mu, var):
    probs = model.mog_pdf(y, pi, mu, var)
    nll = -torch.log(probs)
    return torch.mean(nll)


def coral_loss(model, )


def get_loader(X, theta, batch_size):
    # train / validation split (90 / 10)
    X_train, X_val, theta_train, theta_val = train_test_split(X, theta, test_size=0.1, random_state=7)
    # train loader
    X_train, theta_train = torch.Tensor(X_train), torch.Tensor(theta_train)
    dataset_train = TensorDataset(X_train, theta_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    # val loader
    X_val, theta_val = torch.Tensor(X_val), torch.Tensor(theta_val)
    dataset_val = TensorDataset(X_val, theta_val)
    val_loader = DataLoader(dataset_val, batch_size=X_val.shape[0], shuffle=True)
    return train_loader, val_loader


def train_mdn(model, X, theta, batch_size, lr, n_epoch, verbose=1):
    train_loader, val_loader = get_loader(X, theta, batch_size)
    train_size = len(train_loader.dataset.tensors[0])

    train_loss = np.zeros(n_epoch)
    val_loss = np.zeros(n_epoch)
    best_val_loss = np.inf

    """if not isinstance(model.encoder, type(None)):
        model.encoder.weight.requires_grad = False
        model.encoder.bias.requires_grad = False"""
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(n_epoch):
        for data, theta in train_loader:
            optimizer.zero_grad()
            pi, mu, var = model(data) # var = sigma or inv_sigma
            loss = mdn_loss(model, theta, pi, mu, var)
            train_loss[epoch] += loss.item()
            loss.backward()
            optimizer.step()

        # train loss
        train_loss[epoch] /= train_size // batch_size + 1

        # saves model with lowest validation error
        with torch.no_grad():
            for data, theta in val_loader:
                pi, mu, var = model(data)  # var = sigma or inv_sigma
                val_loss[epoch] = mdn_loss(model, theta, pi, mu, var).item()

        if val_loss[epoch] < best_val_loss:
            torch.save(model, "C:/workspace/likelihood-free-inference/exps/temp_best_model.pb")
            best_val_loss = val_loss[epoch]

        # monitoring
        if verbose == 1:
            print("[%d] Train loss = %f" % (epoch + 1, train_loss[epoch]))
            if epoch + 1 < 10:
                print("    Validation loss = %f" % val_loss[epoch])
            else:
                print("     Validation loss = %f" % val_loss[epoch])
            print("\t -------")

    model.fitted = True
    return torch.load("C:/workspace/likelihood-free-inference/exps/temp_best_model.pb")