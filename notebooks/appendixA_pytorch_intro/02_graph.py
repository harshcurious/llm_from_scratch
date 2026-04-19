import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from  torch.autograd import grad

    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1*w1 + b
    a = torch.sigmoid(z)
    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    return b, grad_L_b, grad_L_w1, loss, w1


@app.cell
def _(grad_L_b, grad_L_w1):
    print(grad_L_w1)
    print(grad_L_b)
    return


@app.cell
def _(b, loss, w1):
    loss.backward()
    print(w1.grad)
    print(b.grad)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
