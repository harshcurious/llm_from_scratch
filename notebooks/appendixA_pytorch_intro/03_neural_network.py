import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


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
def _(NeuralNetwork):
    model = NeuralNetwork(50, 3)
    print(model)
    return (model,)


@app.cell
def _(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", num_params)
    return


@app.cell
def _(model):
    print(model.layers[0].weight)
    return


@app.cell
def _(model):
    print(model.layers[0].weight.shape)
    return


@app.cell
def _(model):
    print(model.layers[0].bias.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
