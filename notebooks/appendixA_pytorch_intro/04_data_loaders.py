import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _(torch):
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5],
        # [3.1, 7]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    from torch.utils.data import Dataset

    class ToyDataset(Dataset):
        def __init__(self, X, y):
            self.features = X
            self.labels = y

        def __getitem__(self, index):
            one_x = self.features[index]
            one_y = self.labels[index]
            return one_x, one_y

        def __len__(self):
            return self.labels.shape[0]

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)
    return test_ds, train_ds


@app.cell
def _(torch):
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, num_inputs, num_outputs):
            super().__init__()

            self.layers = torch.nn.Sequential(
                # 1st hidden layer
                torch.nn.Linear(num_inputs, 30),
                torch.nn.ReLU(),

                # 2nd hidden layer
                torch.nn.Linear(30, 20),
                torch.nn.ReLU(),

                # output layer
                torch.nn.Linear(20, num_outputs),
            )

        def forward(self, x):
            logits = self.layers(x)
            return logits

    return (NeuralNetwork,)


@app.cell
def _(test_ds, torch, train_ds):
    from torch.utils.data import DataLoader

    torch.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=2, shuffle=True, num_workers=0
    )
    return (train_loader,)


@app.cell
def _(train_loader):
    for i, (x,y) in enumerate(train_loader):
        print(f"Batch {i + 1}:", x, y)
    return


@app.cell
def _(NeuralNetwork, torch, train_loader):
    import torch.nn.functional as F

    torch.manual_seed(42)

    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    print(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epoch = 3

    for epoch in range(num_epoch):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
